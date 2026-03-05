import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from oncolearn.registry.modalities import register_modality
from oncolearn.api.xenabrowser.builder import XenaCohortBuilder
from oncolearn.data.modalities.tabular.parsers import DEFAULT_PARSERS


class TabularDataset(Dataset):
    """
    Internal PyTorch Dataset for tabular features (e.g. gene expressions).
    Derived structurally from the underlying DataFrame.
    """
    def __init__(self, df: pd.DataFrame, patient_id_col: str = "patient_id", label_col: Optional[str] = None):
        """
        Args:
            df: The pure dataframe post-parsing.
            patient_id_col: Column name denoting the patient ID index.
            label_col: Optional column name for labels.
        """
        self.df = df
        self.patient_id_col = patient_id_col
        self.label_col = label_col
        
        if self.patient_id_col not in self.df.columns:
            raise KeyError(f"Expected patient ID column '{self.patient_id_col}' not found in dataframe columns: {self.df.columns.tolist()[:5]}...")
            
        # Extract metadata
        raw_ids = self.df[self.patient_id_col].values.tolist()
        self.patient_ids = [
            pid[:12] if isinstance(pid, str) and pid.startswith("TCGA-") else pid 
            for pid in raw_ids
        ]
        
        # Identify feature columns (everything except ID and potentially label)
        exclude = [self.patient_id_col]
        if self.label_col and self.label_col in self.df.columns:
            exclude.append(self.label_col)
            
        self.feature_cols = [c for c in self.df.columns if c not in exclude]
        
        # Pre-convert to float matrix for fast tensor extraction
        # Fill NA with 0 by default for gene expressions
        self.features_matrix = self.df[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float).values
        
        if self.label_col and self.label_col in self.df.columns:
            self.labels = self.df[self.label_col].values
        else:
            self.labels = None

    def get_keys(self) -> List[str]:
        """Method required by MultimodalDataset to align records."""
        return self.patient_ids
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        features = torch.tensor(self.features_matrix[idx], dtype=torch.float32)
        patient_id = self.patient_ids[idx]
        
        result = {
            "tabular": features,
            "patient_id": patient_id
        }
        
        if self.labels is not None:
            # Simple conversion, assumes integer classes. 
            # In a real scenario we'd want a LabelEncoder fit on the whole dataset.
            try:
                result["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            except ValueError:
                result["label"] = self.labels[idx]
                
        return result


@register_modality("tabular")
class TabularDataModule(pl.LightningDataModule):
    """
    API-first LightningDataModule for Tabular Data.
    Uses XenaCohortBuilder to grab datasets from Xenabrowser.
    """
    def __init__(
        self,
        cohort_code: str = "TCGA-BRCA",
        batch_size: int = 16,
        num_workers: int = 4,
        data_dir: str = "data/xenabrowser",
        train_split: float = 0.8,
        seed: int = 42,
        label_column: Optional[str] = None,
        features_files: Optional[List[str]] = None
    ):
        # Default to miRNA + PAM50 label for TCGA-BRCA.
        # miRNA (1881 features) is within sequence-length limits for the gene encoder.
        if features_files is None and cohort_code == "TCGA-BRCA":
            features_files = ["TCGA-BRCA.mirna.tsv", "pam50.tsv"]
            
        super().__init__()
        self.name = "tabular"
        self.cohort_code = cohort_code
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.seed = seed
        self.label_column = label_column
        self.features_files = features_files
        
        self.builder = XenaCohortBuilder()
        self.cohort_api = None
        self.master_df = None

    def prepare_data(self):
        """
        Download required Xenabrowser cohort matrices, skipping if data already exists.
        """
        cohort_dir = self.data_dir / self.cohort_code
        if cohort_dir.exists() and any(cohort_dir.rglob("*.tsv")):
            print(f"Tabular data already present in {cohort_dir}, skipping download.")
            return
        try:
            self.cohort_api = self.builder.build_cohort(self.cohort_code)
            self.cohort_api.download(output_dir=str(cohort_dir), download_all=True)
        except Exception as e:
            print(f"Error preparing tabular data API: {e}")

    def setup(self, stage: Optional[str] = None):
        """
        Parse downloaded files, merge them, and build the PyTorch Dataset.
        """
        cohort_dir = self.data_dir / self.cohort_code
        if not cohort_dir.exists():
             print(f"Warning: Cohort directory {cohort_dir} not found. Ensure prepare_data() succeeded.")
             self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
             return
             
        # Parse all tabular files in the directory
        dfs = []
        
        # If specific files are requested, use those. Otherwise all files.
        if self.features_files:
            targets = [cohort_dir / f for f in self.features_files]
        else:
            targets = list(cohort_dir.rglob("*"))
            
        for file_path in targets:
            if file_path.is_file():
                for parser in DEFAULT_PARSERS:
                    if parser.can_parse(file_path):
                        df = parser.parse(file_path)
                        dfs.append(df)
                        break
                        
        if not dfs:
            print(f"No parseable tabular files found in {cohort_dir}")
            self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
            return
            
        print(f"TabularDataModule loaded {len(dfs)} matrices. Merging...")
        
        # Merge all dataframes on 'patient_id' (inner join for intersection)
        self.master_df = dfs[0]
        for df in dfs[1:]:
            self.master_df = pd.merge(self.master_df, df, on="patient_id", how="inner")
            
        print(f"Merged tabular representation has {len(self.master_df)} patients.")

        # Auto-detect label column if not explicitly set
        label_col = self.label_column
        if label_col is None and "label" in self.master_df.columns:
            label_col = "label"

        # Split conceptually via dataframe shuffle instead of PyTorch Subset
        total_size = len(self.master_df)
        train_size = int(self.train_split * total_size)

        shuffled_df = self.master_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        train_df = shuffled_df.iloc[:train_size]
        val_df = shuffled_df.iloc[train_size:]

        self._full_dataset = TabularDataset(self.master_df, label_col=label_col)
        self.train_dataset = TabularDataset(train_df, label_col=label_col)
        self.val_dataset = TabularDataset(val_df, label_col=label_col)
        self.test_dataset = self.val_dataset

    @property
    def full_dataset(self) -> "TabularDataset":
        """Full dataset (all patients, no split) — available after setup()."""
        return self._full_dataset

    def setup_full(self, stage=None):
        """Ensure setup() has been called so full_dataset is available."""
        if not hasattr(self, "_full_dataset"):
            self.setup(stage=stage)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
