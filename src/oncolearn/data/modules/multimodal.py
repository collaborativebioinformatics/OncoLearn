from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from oncolearn.registry.modalities import get_modality
from oncolearn.registry.datasets import register_dataset
import oncolearn.data.modalities  # noqa: F401 — triggers @register_modality decorators
from oncolearn.cli.utils.splits import read_id_file
from oncolearn.data.modalities.multimodal import MultimodalDataset
from .base import OncoDataModule


def _load_split_ids(splits_dir: str) -> Dict[str, Optional[Set[str]]]:
    """Load patient IDs from split files in *splits_dir*.

    Expects ``train.txt``, ``test.txt``, ``validation.txt`` (one ID per line).
    Falls back: missing validation.txt → use test IDs; missing test.txt → use val IDs.
    """
    p = Path(splits_dir)
    train_ids = read_id_file(p / "train.txt")
    test_ids = read_id_file(p / "test.txt")
    val_ids = read_id_file(p / "validation.txt")
    if val_ids is None and test_ids is not None:
        val_ids = test_ids  # validation.txt missing — use test split
    elif test_ids is None and val_ids is not None:
        test_ids = val_ids  # test.txt missing — use validation split
    return {"train": train_ids, "val": val_ids, "test": test_ids}


@register_dataset("oncolearn.datasets.multimodal")
class MultimodalDataModule(OncoDataModule):
    """
    A PyTorch Lightning builder that accepts either string registry names
    or instantiated DataModules, and merges them for multimodal training.
    """
    def __init__(
        self,
        modalities: List[Union[str, pl.LightningDataModule]],
        join_on: str = "patient_id",
        strategy: str = "inner",
        batch_size: int = 16,
        num_workers: int = 4,
        splits_dir: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            modalities: List of modality names or DataModules.
            join_on: Identifier key to merge patients.
            strategy: Inner join vs outer join.
            batch_size: DataLoader batch size.
            num_workers: DataLoader num workers.
            splits_dir: Path to folder with train/test/validation.txt split files.
                        If provided, split files override per-modality random splits.
        """
        super().__init__()
        self.name = "multimodal"
        self.join_on = join_on
        self.strategy = strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits_dir = splits_dir
        self.num_classes = num_classes
        self.class_weights: Optional[torch.Tensor] = None

        self.datamodules = {}
        for mod in modalities:
            if isinstance(mod, str):
                mod_cls = get_modality(mod)
                dm_instance = mod_cls()
                self.datamodules[mod] = dm_instance
            elif isinstance(mod, pl.LightningDataModule):
                name = getattr(mod, "name", mod.__class__.__name__.lower())
                self.datamodules[name] = mod
            else:
                raise ValueError(f"Unknown modality type: {type(mod)}")

    def prepare_data(self):
        for name, dm in self.datamodules.items():
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.splits_dir:
            # External split files — build full datasets then filter by ID
            for name, dm in self.datamodules.items():
                dm.setup_full(stage=stage)

            splits = _load_split_ids(self.splits_dir)
            full_datasets = {
                name: dm.full_dataset
                for name, dm in self.datamodules.items()
                if dm.full_dataset is not None
            }

            if stage == "fit" or stage is None:
                self.train_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["train"],
                )
                self.val_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["val"],
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["test"],
                )
        else:
            # Default: per-modality random splits
            for name, dm in self.datamodules.items():
                dm.setup(stage=stage)

            if stage == "fit" or stage is None:
                self.train_dataset = MultimodalDataset(
                    {name: dm.train_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )
                self.val_dataset = MultimodalDataset(
                    {name: dm.val_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultimodalDataset(
                    {name: dm.test_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )

        # Compute inverse-frequency class weights from the training split
        if (stage == "fit" or stage is None) and hasattr(self, "train_dataset"):
            self._compute_class_weights()

    def setup_full(self, stage: Optional[str] = None) -> None:
        """Not applicable for MultimodalDataModule — delegates to setup()."""
        self.setup(stage=stage)

    @property
    def full_dataset(self):
        """Not applicable for MultimodalDataModule — returns train_dataset."""
        return getattr(self, "train_dataset", None)

    def _compute_class_weights(self) -> None:
        valid_labels = [l for l in self.train_dataset.get_labels() if l is not None]
        if not valid_labels:
            return
        counts = Counter(valid_labels)
        n_classes = self.num_classes if self.num_classes is not None else max(counts) + 1
        total = len(valid_labels)
        self.class_weights = torch.tensor(
            [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
            dtype=torch.float32,
        )
