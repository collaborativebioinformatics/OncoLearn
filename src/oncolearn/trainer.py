"""
OncoLearn training orchestrator.

Typical usage::

    from oncolearn.config import load_config
    from oncolearn.trainer import OncoTrainer

    config = load_config("data/configs/tcga_brca_tabular_only.yaml")
    trainer = OncoTrainer(config)
    trainer.train()
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl

# Utilize Tensor Cores on supported GPUs (trades negligible precision for performance)
torch.set_float32_matmul_precision("high")
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from oncolearn.config import OncoLearnConfig, load_config
from oncolearn.registry import get_model, get_modality
import oncolearn.modeling  # noqa: F401 — triggers @register_model / @register_encoder decorators
from oncolearn.data.multimodal import MultimodalDataModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# OncoTrainer
# ---------------------------------------------------------------------------

class OncoTrainer:
    """Config-driven training orchestrator for OncoLearn experiments.

    The trainer is fully specified by an :class:`~oncolearn.config.OncoLearnConfig`:

    1. The model is looked up in the model registry by ``config.model.name`` and
       instantiated directly with the config.
    2. Each data modality is looked up in the modality registry by name and
       initialised with its per-modality kwargs from the config.
    3. A :class:`pytorch_lightning.Trainer` is configured from ``config.training``
       and drives the full training loop via the Lightning module.
    4. Checkpoints are written to ``config.output.dir / config.output.experiment_name``.

    Example::

        config = load_config("data/configs/tcga_brca_tabular_only.yaml")
        trainer = OncoTrainer(config)
        best_metrics = trainer.train()
        test_metrics = trainer.test()
    """

    def __init__(self, config: OncoLearnConfig) -> None:
        self.config = config
        set_seed(config.training.seed)

        self.device = self._resolve_device()
        self.datamodule = self._build_datamodule()
        self.model = self._build_model()

        self._pl_trainer: Optional[pl.Trainer] = None

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        acc = self.config.training.accelerator
        if acc == "cpu":
            return torch.device("cpu")
        if acc == "cuda" or (acc == "auto" and torch.cuda.is_available()):
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_datamodule(self) -> MultimodalDataModule:
        """Instantiate each registered modality DataModule with its config kwargs."""
        dm_instances = []
        for mod_cfg in self.config.modalities:
            dm_cls = get_modality(mod_cfg.name)
            dm = dm_cls(**mod_cfg.kwargs)
            dm.name = mod_cfg.name  # required by MultimodalDataModule
            dm_instances.append(dm)

        t = self.config.training
        return MultimodalDataModule(
            modalities=dm_instances,
            join_on=self.config.join_on,
            strategy=self.config.join_strategy,
            batch_size=t.batch_size,
            num_workers=t.num_workers,
            splits_dir=self.config.splits_dir,
            num_classes=self.config.model.num_stage_classes,
        )

    def _build_model(self) -> pl.LightningModule:
        """Look up the registered model class and instantiate it with the config."""
        model_cls = get_model(self.config.model.name)
        return model_cls(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop via PyTorch Lightning.

        Returns:
            Final callback metrics from the Lightning trainer.
        """
        t = self.config.training
        out = self.config.output
        output_dir = Path(out.dir) / out.experiment_name

        callbacks = [
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="best_model",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
            ),
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="epoch_{epoch}",
                every_n_epochs=out.save_every_n_epochs,
                save_top_k=-1,
            ),
            EarlyStopping(
                monitor="val_acc",
                patience=t.early_stopping_patience,
                mode="max",
            ),
        ]

        self._pl_trainer = pl.Trainer(
            max_epochs=t.max_epochs,
            accelerator=t.accelerator,
            devices=t.devices,
            default_root_dir=str(output_dir),
            callbacks=callbacks,
            log_every_n_steps=1,
        )

        logger.info(
            "Training | model=%s | modalities=%s | device=%s | epochs=%d",
            self.config.model.name,
            [m.name for m in self.config.modalities],
            self.device,
            t.max_epochs,
        )

        self._pl_trainer.fit(self.model, datamodule=self.datamodule)
        return dict(self._pl_trainer.callback_metrics)

    def test(self) -> Dict[str, float]:
        """Evaluate on the test split.

        Returns:
            Test metrics dict.
        """
        if self._pl_trainer is None:
            raise RuntimeError("Call train() before test()")
        results = self._pl_trainer.test(self.model, datamodule=self.datamodule)
        return results[0] if results else {}

    def save_checkpoint(self, path: Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model state dict from *path*."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        logger.info("Loaded checkpoint from %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m oncolearn.trainer``.

    Delegates to :func:`oncolearn.cli.train.main` for the full argument set.

    Examples::

        python -m oncolearn.trainer --config data/configs/tcga_brca_tabular_only.yaml
        python -m oncolearn.trainer --variant v2_no_imaging --epochs 10 --batch_size 8
    """
    from oncolearn.cli.train import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
