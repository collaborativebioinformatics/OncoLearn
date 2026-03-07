"""
Shared TabularDataset used by GeneDataModule and ClinicalDataModule.
"""
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Generic PyTorch Dataset for tabular features derived from a DataFrame.

    All columns except ``patient_id_col`` and ``label_col`` are coerced to
    float32, with NaN filled as 0.  Used by both GeneDataModule and
    ClinicalDataModule.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patient_id_col: str = "patient_id",
        label_col: Optional[str] = None,
        batch_key: str = "tabular",
    ):
        """
        Args:
            df: DataFrame post-parsing.
            patient_id_col: Column name for patient IDs.
            label_col: Optional column name for integer class labels.
            batch_key: Key used in the returned dict (e.g. ``"tabular"`` or
                       ``"clinical"``).  Must match the encoder name in the
                       fusion model config.
        """
        self.batch_key = batch_key

        if patient_id_col not in df.columns:
            raise KeyError(
                f"Expected patient ID column '{patient_id_col}' not found. "
                f"Available: {df.columns.tolist()[:5]}..."
            )

        raw_ids = df[patient_id_col].values.tolist()
        self.patient_ids = [
            pid[:12] if isinstance(pid, str) and pid.startswith("TCGA-") else pid
            for pid in raw_ids
        ]

        exclude = {patient_id_col}
        if label_col and label_col in df.columns:
            exclude.add(label_col)

        feat_cols = [c for c in df.columns if c not in exclude]
        self.features_matrix = (
            df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
        )
        self.feature_dim: int = self.features_matrix.shape[1]

        self.labels = df[label_col].values if (label_col and label_col in df.columns) else None

    def get_keys(self) -> List[str]:
        return self.patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            self.batch_key: torch.tensor(self.features_matrix[idx], dtype=torch.float32),
            "patient_id": self.patient_ids[idx],
        }
        if self.labels is not None:
            try:
                result["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            except (ValueError, TypeError):
                result["label"] = self.labels[idx]
        return result
