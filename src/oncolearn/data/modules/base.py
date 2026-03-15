"""
OncoDataModule: abstract base class for all OncoLearn LightningDataModules.
"""
from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class OncoDataModule(pl.LightningDataModule, ABC):
    """Abstract base for all OncoLearn data modules.

    Subclasses must implement :meth:`setup` and :meth:`setup_full`, and expose
    ``train_dataset``, ``val_dataset``, ``test_dataset``, and ``full_dataset``
    after those methods are called.

    Concrete :meth:`train_dataloader`, :meth:`val_dataloader`, and
    :meth:`test_dataloader` are provided here using ``self.batch_size`` and
    ``self.num_workers`` which subclasses must set in ``__init__``.
    """

    # Subclasses must set these in __init__
    batch_size: int
    num_workers: int
    name: str

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Build train/val/test splits and populate ``*_dataset`` attributes."""

    @abstractmethod
    def setup_full(self, stage: Optional[str] = None) -> None:
        """Build the full dataset without splitting (for multimodal alignment)."""

    @property
    @abstractmethod
    def full_dataset(self):
        """Return the full (unsplit) dataset after :meth:`setup_full` is called."""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
