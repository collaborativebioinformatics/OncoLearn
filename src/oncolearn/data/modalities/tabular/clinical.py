"""
Clinical tabular data modality from XenaBrowser TCGA clinical TSV files.
"""
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oncolearn.registry.modalities import register_modality
from .base import TabularDataset
from .parsers import ClinicalParser


@register_modality("clinical", "oncolearn.modality.clinical")
class ClinicalDataModule(pl.LightningDataModule):
    """
    LightningDataModule for numeric clinical features (TCGA-BRCA.clinical.tsv).

    Uses :class:`ClinicalParser` to extract numeric-only columns and map AJCC
    pathologic stage to integer labels.

    Args:
        cohort_code: TCGA cohort identifier (e.g. ``"TCGA-BRCA"``).
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        base_directory: Root directory for Xena data.  Alias for ``data_dir``.
        data_dir: Deprecated alias for ``base_directory``.
        train_split: Fraction of data to use for training.
        seed: Random seed for reproducible splits.
        clinical_file: TSV file name.  Overridden by ``files[0]`` when ``files``
                       is provided.
        files: List of file names; ``files[0]`` is used as the clinical TSV.
        stage_col: Column name containing AJCC pathologic stage.
        batch_key: Key used in the batch dict (defaults to ``"clinical"``).
    """

    def __init__(
        self,
        cohort_code: str = "TCGA-BRCA",
        batch_size: int = 16,
        num_workers: int = 4,
        base_directory: Optional[str] = None,
        data_dir: str = "data/xenabrowser",
        train_split: float = 0.8,
        seed: int = 42,
        clinical_file: str = "TCGA-BRCA.clinical.tsv",
        files: Optional[List[str]] = None,
        stage_col: str = ClinicalParser.STAGE_COL,
        batch_key: str = "clinical",
    ):
        # Resolve aliases
        resolved_dir = base_directory if base_directory is not None else data_dir
        resolved_file = files[0] if files else clinical_file

        super().__init__()
        self.name = "clinical"
        self.cohort_code = cohort_code
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(resolved_dir)
        self.train_split = train_split
        self.seed = seed
        self.clinical_file = resolved_file
        self.stage_col = stage_col
        self.batch_key = batch_key
        self._full_dataset: Optional["TabularDataset"] = None

    def prepare_data(self):
        # Clinical TSV is downloaded alongside gene data by XenaCohortBuilder.
        pass

    def _load_df(self):
        file_path = self.data_dir / self.cohort_code / self.clinical_file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Clinical file not found: {file_path}. "
                "Ensure gene modality prepare_data() has run first."
            )
        df = ClinicalParser.parse(file_path, stage_col=self.stage_col)
        print(
            f"ClinicalDataModule: {len(df)} patients, "
            f"{df.shape[1] - 2} numeric features."  # -2 for patient_id + label
        )
        return df

    def setup(self, stage: Optional[str] = None):
        try:
            df = self._load_df()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            self.train_dataset = self.val_dataset = self.test_dataset = []
            return

        label_col = "label" if "label" in df.columns else None
        self._full_dataset = TabularDataset(df, label_col=label_col, batch_key=self.batch_key)

        total = len(df)
        train_size = int(self.train_split * total)
        shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.train_dataset = TabularDataset(
            shuffled.iloc[:train_size], label_col=label_col, batch_key=self.batch_key
        )
        self.val_dataset = TabularDataset(
            shuffled.iloc[train_size:], label_col=label_col, batch_key=self.batch_key
        )
        self.test_dataset = self.val_dataset

    @property
    def full_dataset(self) -> TabularDataset:
        return self._full_dataset

    def setup_full(self, stage=None):
        if not hasattr(self, "_full_dataset"):
            self.setup(stage=stage)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )
