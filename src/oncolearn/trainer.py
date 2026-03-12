"""
OncoLearn training orchestrator.

Typical usage::

    from oncolearn.config import load_config
    from oncolearn.trainer import OncoTrainer

    config = load_config("data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml")
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from oncolearn.config import OncoLearnConfig, load_config
from oncolearn.registry import get_model, get_dataset
import oncolearn.modeling  # noqa: F401 — triggers @register_model / @register_encoder decorators
import oncolearn.data.modules  # noqa: F401 — triggers @register_dataset decorators
from oncolearn.data.modules.multimodal import MultimodalDataModule

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

        config = load_config("data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml")
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
        """Load the pipeline file and instantiate a datamodule per modality."""
        from oncolearn.data.pipeline.loader import load_pipeline_file
        from oncolearn.data.modules.tabular import TabularDataModule
        from oncolearn.data.pipeline.nodes import ImageModality, TabularModality

        data_cfg = self.config.data
        t = self.config.training

        dataset_node = load_pipeline_file(data_cfg.pipeline)
        dms = []
        for m in dataset_node.modalities:
            if isinstance(m, ImageModality):
                from oncolearn.data.modules.image import ImageDataModule
                dm = ImageDataModule(
                    cohort_code=m.cohort_code,
                    base_directory=m.base_dir,
                    n_slices=m.n_slices,
                    prefer_mr=m.prefer_mr,
                    batch_size=t.batch_size,
                    num_workers=t.num_workers,
                    seed=t.seed,
                    batch_key=m.name,
                )
                dm.name = m.name
                dms.append(dm)
            elif isinstance(m, TabularModality):
                dms.append(TabularDataModule.from_modality(
                    m,
                    batch_size=t.batch_size,
                    num_workers=t.num_workers,
                    seed=t.seed,
                ))
            else:
                raise TypeError(
                    f"Unknown modality node type: {type(m)}. "
                    f"Expected TabularModality or ImageModality."
                )

        # Resolve dataset class: registry lookup if dataset_node.name is set,
        # else fall back to MultimodalDataModule.
        if dataset_node.name:
            dataset_cls = get_dataset(dataset_node.name)
        else:
            dataset_cls = MultimodalDataModule

        return dataset_cls(
            modalities=dms,
            join_on=dataset_node.join_on,
            strategy=dataset_node.join_strategy,
            batch_size=t.batch_size,
            num_workers=t.num_workers,
            splits_dir=data_cfg.splits_dir,
            num_classes=self.config.model.num_stage_classes,
        )

    def _build_model(self) -> pl.LightningModule:
        """Look up the registered model class and instantiate it with the config."""
        model_cls = get_model(self.config.model.name)
        self._check_modality_compatibility(model_cls)
        return model_cls(self.config)

    def _check_modality_compatibility(self, model_cls) -> None:
        """No-op: modality validation is now deferred to pipeline execution."""
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop via PyTorch Lightning.

        If ``config.training.hpo`` is set, an Optuna study is run first and
        the best-found hyperparameters are applied to the config before the
        final training run begins.

        Returns:
            Final callback metrics from the Lightning trainer.
        """
        if self.config.training.hpo is not None:
            self._run_hpo()

        t = self.config.training
        out = self.config.output
        output_dir = Path(out.dir) / out.experiment_name

        callbacks = [
            ModelSummary(max_depth=2),
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

        gradient_clip_val = t.regularization.gradient_clip_val if t.regularization.gradient_clip_val > 0 else None
        tb_logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="tensorboard",
        )
        self._pl_trainer = pl.Trainer(
            max_epochs=t.max_epochs,
            accelerator=t.accelerator,
            devices=t.devices,
            default_root_dir=str(output_dir),
            callbacks=callbacks,
            logger=tb_logger,
            log_every_n_steps=1,
            gradient_clip_val=gradient_clip_val,
        )

        logger.info(
            "Training | model=%s | pipeline=%s | device=%s | epochs=%d",
            self.config.model.name,
            self.config.data.pipeline,
            self.device,
            t.max_epochs,
        )

        self._pl_trainer.fit(self.model, datamodule=self.datamodule)
        return dict(self._pl_trainer.callback_metrics)

    def _run_hpo(self) -> None:
        """Run an Optuna study and apply best params to self.config in-place."""
        from oncolearn.modeling.hyps import OptunaHPTuner

        hpo_cfg = self.config.training.hpo
        logger.info(
            "HPO enabled — running %d trials before final training", hpo_cfg.n_trials
        )
        tuner = OptunaHPTuner(self.config, hpo_cfg)
        best_params, best_config = tuner.tune()

        # Apply best params back to this trainer's live config so the final
        # training run (which follows immediately) uses them.
        from oncolearn.modeling.hyps import apply_params
        apply_params(self.config, best_params)

        # Rebuild model/datamodule with updated config
        self.datamodule = self._build_datamodule()
        self.model = self._build_model()

        logger.info("HPO complete — best params applied: %s", best_params)

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

    Delegates to :func:`oncolearn.cli.subcommands.train.command.main` for the full argument set.

    Examples::

        python -m oncolearn.trainer --config data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml
        python -m oncolearn.trainer --variant v2_no_imaging --epochs 10 --batch_size 8
    """
    from oncolearn.cli.subcommands.train.command import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
