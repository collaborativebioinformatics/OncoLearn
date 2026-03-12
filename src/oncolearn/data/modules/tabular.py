"""
PipelineDataModule: LightningDataModule backed by the pipeline DSL.
"""
from typing import Optional

from torch.utils.data import DataLoader

from oncolearn.data.pipeline.nodes import TabularModality
from oncolearn.data.pipeline.executor import run
from oncolearn.data.pipeline.loader import load_pipeline_file, _make_reader  # re-export for convenience
from oncolearn.data.modalities.tabular import TabularDataset
from .base import OncoDataModule


class PipelineDataModule(OncoDataModule):
    """LightningDataModule that loads data via the pipeline DSL.

    Executes the modality's pipeline to produce a DataFrame, applies label
    transforms, wraps in a :class:`TabularDataset`, and splits into
    train/val/test.

    Args:
        modality: The :class:`Modality` descriptor.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        train_split: Fraction of patients for training (remainder split 50/50
                     between val and test).
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        modality: TabularModality,
        batch_size: int = 16,
        num_workers: int = 4,
        train_split: float = 0.8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.name = modality.name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self._full_dataset: Optional[TabularDataset] = None

    @classmethod
    def from_modality(
        cls,
        modality: TabularModality,
        batch_size: int = 16,
        num_workers: int = 4,
        train_split: float = 0.8,
        seed: int = 42,
    ) -> "PipelineDataModule":
        """Factory: create a PipelineDataModule from a Modality node."""
        return cls(
            modality=modality,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
            seed=seed,
        )

    def _load_df(self):
        """Execute the pipeline and apply label transforms.  Returns a DataFrame."""
        reader = _make_reader(self.modality)
        df = run(self.modality.pipeline, reader)

        label_col = self.modality.label_col
        if label_col and label_col in df.columns:
            transform = self.modality.label_transform
            df["label"] = df[label_col].apply(transform) if transform else df[label_col]
            df = df[df["label"].notna()].copy()
            df["label"] = df["label"].astype(int)
            df.drop(columns=[label_col], inplace=True)

        return df

    def setup(self, stage: Optional[str] = None) -> None:
        """Execute the pipeline and build train/val/test splits."""
        df = self._load_df()
        pid_col = self.modality.patient_id_col
        label_col = "label" if "label" in df.columns else None

        self._full_dataset = TabularDataset(
            df,
            patient_id_col=pid_col,
            label_col=label_col,
            batch_key=self.modality.name,
        )

        total = len(df)
        train_size = int(self.train_split * total)
        remaining = total - train_size
        val_size = remaining // 2

        shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.train_dataset = TabularDataset(
            shuffled.iloc[:train_size],
            patient_id_col=pid_col,
            label_col=label_col,
            batch_key=self.modality.name,
        )
        self.val_dataset = TabularDataset(
            shuffled.iloc[train_size: train_size + val_size],
            patient_id_col=pid_col,
            label_col=label_col,
            batch_key=self.modality.name,
        )
        self.test_dataset = TabularDataset(
            shuffled.iloc[train_size + val_size:],
            patient_id_col=pid_col,
            label_col=label_col,
            batch_key=self.modality.name,
        )

    def setup_full(self, stage: Optional[str] = None) -> None:
        """Load the full dataset without splitting (used by MultimodalDataModule)."""
        if self._full_dataset is None:
            df = self._load_df()
            pid_col = self.modality.patient_id_col
            label_col = "label" if "label" in df.columns else None
            self._full_dataset = TabularDataset(
                df,
                patient_id_col=pid_col,
                label_col=label_col,
                batch_key=self.modality.name,
            )

    @property
    def full_dataset(self) -> Optional[TabularDataset]:
        return self._full_dataset


# Alias kept for clarity in imports
TabularDataModule = PipelineDataModule
