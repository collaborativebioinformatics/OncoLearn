"""
Gene expression data modality (miRNA, RNA-seq, etc.) from XenaBrowser.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oncolearn.registry.modalities import register_modality
from oncolearn.api.xenabrowser.builder import XenaCohortBuilder
from .base import TabularDataset
from .parsers import GeneParser


@register_modality("gene", "oncolearn.modality.gene")
class GeneDataModule(pl.LightningDataModule):
    """
    LightningDataModule for gene expression / miRNA tabular features.

    Downloads TCGA cohort matrices from XenaBrowser, merges them on
    ``patient_id`` (inner join), and wraps the result in a
    :class:`TabularDataset`.

    Args:
        cohort_code: TCGA cohort identifier (e.g. ``"TCGA-BRCA"``).
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        base_directory: Root directory for Xena data.  Alias for ``data_dir``.
        data_dir: Deprecated alias for ``base_directory``.
        train_split: Fraction of data to use for training (random split).
        seed: Random seed for reproducible splits.
        label_column: Name of the label column.  Auto-detected when ``None``.
        files: List of TSV file names to load.  Alias for ``features_files``.
        features_files: Deprecated alias for ``files``.
        batch_key: Key used in the batch dict (defaults to ``"gene"`` for
                   backward compatibility; pass ``mod_cfg.name`` for dotted-name routing).
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
        label_column: Optional[str] = None,
        files: Optional[List[str]] = None,
        features_files: Optional[List[str]] = None,
        batch_key: str = "gene",
    ):
        # Resolve aliases: base_directory > data_dir
        resolved_dir = base_directory if base_directory is not None else data_dir
        # Resolve aliases: files > features_files > default
        resolved_files = files if files is not None else features_files
        if resolved_files is None and cohort_code == "TCGA-BRCA":
            resolved_files = ["TCGA-BRCA.mirna.tsv", "pam50.tsv"]

        super().__init__()
        self.name = "gene"
        self.cohort_code = cohort_code
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(resolved_dir)
        self.train_split = train_split
        self.seed = seed
        self.label_column = label_column
        self.features_files = resolved_files
        self.batch_key = batch_key

        self.builder = XenaCohortBuilder()

    def prepare_data(self):
        cohort_dir = self.data_dir / self.cohort_code
        if cohort_dir.exists() and any(cohort_dir.rglob("*.tsv")):
            print(f"Gene data already present in {cohort_dir}, skipping download.")
            return
        try:
            cohort_api = self.builder.build_cohort(self.cohort_code)
            cohort_api.download(output_dir=str(cohort_dir), download_all=True)
        except Exception as e:
            print(f"Error downloading gene data: {e}")

    def _load_df(self) -> pd.DataFrame:
        cohort_dir = self.data_dir / self.cohort_code
        targets = (
            [cohort_dir / f for f in self.features_files]
            if self.features_files
            else list(cohort_dir.rglob("*"))
        )
        dfs = []
        for file_path in targets:
            if file_path.is_file() and GeneParser.can_parse(file_path):
                dfs.append(GeneParser.parse(file_path))

        if not dfs:
            raise RuntimeError(f"No parseable gene expression files found in {cohort_dir}")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="patient_id", how="inner")

        print(f"GeneDataModule: merged {len(merged)} patients from {len(dfs)} file(s).")
        return merged

    def setup(self, stage: Optional[str] = None):
        try:
            df = self._load_df()
        except Exception as e:
            print(f"Warning: {e}")
            self.train_dataset = self.val_dataset = self.test_dataset = []
            self._full_dataset = None
            return

        label_col = self.label_column
        if label_col is None and "label" in df.columns:
            label_col = "label"

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
        """Load all patients without label filtering for use as a feature encoder.

        Unlike ``setup()``, this method keeps patients that lack PAM50 labels so
        that the gene modality covers the full multimodal intersection (e.g. all
        57 imaging patients even when only 6 of them have PAM50 annotations).
        Labels in the full dataset are omitted; the clinical modality supplies
        stage labels during multimodal training.
        """
        if not hasattr(self, "_full_dataset"):
            from .parsers.xenabrowser_parser import XenabrowserParser

            cohort_dir = self.data_dir / self.cohort_code
            targets = (
                [cohort_dir / f for f in self.features_files]
                if self.features_files
                else list(cohort_dir.rglob("*"))
            )
            dfs = []
            for file_path in targets:
                if file_path.is_file() and XenabrowserParser.can_parse(file_path):
                    dfs.append(XenabrowserParser.load(file_path))

            if not dfs:
                self._full_dataset = None
                return

            merged = dfs[0]
            for df in dfs[1:]:
                merged = pd.merge(merged, df, on="patient_id", how="left")

            print(
                f"GeneDataModule (full): merged {len(merged)} patients"
                f" from {len(dfs)} file(s)."
            )
            # No label_col — labels come from the clinical modality in multimodal mode
            self._full_dataset = TabularDataset(
                merged, label_col=None, batch_key=self.batch_key
            )

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
