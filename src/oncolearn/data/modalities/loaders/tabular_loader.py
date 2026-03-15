"""
Tabular data loaders for OncoLearn.

XenabrowserParser inherits from BaseDataLoader and implements can_load/load
using the standard loader protocol.
"""
from pathlib import Path

import pandas as pd

from .base import BaseDataLoader
from oncolearn.data.utils import normalize_patient_id


class XenabrowserParser(BaseDataLoader):
    """
    XenaBrowser TSV loader implementing the BaseDataLoader protocol.

    :meth:`can_load` checks for .tsv extension.
    :meth:`load` returns a DataFrame with a ``patient_id`` column and feature
    columns, but no label column.
    """

    @classmethod
    def can_load(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".tsv"

    @classmethod
    def _is_genomic_matrix(cls, df: pd.DataFrame) -> bool:
        """Return True if columns (after the first) look like TCGA sample IDs."""
        sample_cols = [
            c for c in df.columns[1:6]
            if isinstance(c, str) and c.startswith("TCGA-")
        ]
        return len(sample_cols) >= 3

    @classmethod
    def load(cls, file_path: Path) -> pd.DataFrame:
        """
        Load a XenaBrowser TSV and return a normalized DataFrame.

        - Genomic matrix files are transposed so rows become patients.
        - Patient IDs are truncated to the standard 12-character TCGA format.
        - No label encoding is performed here.
        """
        try:
            df = pd.read_csv(str(file_path), sep="\t", low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read XenaBrowser TSV at {file_path}: {e}")

        # Normalize layout to rows=patients
        if cls._is_genomic_matrix(df):
            id_col = df.columns[0]
            df = df.set_index(id_col).T.reset_index()
            df = df.rename(columns={"index": "patient_id"})
        elif "sample" in df.columns:
            df = df.rename(columns={"sample": "patient_id"})
        elif df.columns[0] == "Unnamed: 0":
            df = df.rename(columns={"Unnamed: 0": "patient_id"})

        # Truncate to 12-char TCGA patient ID
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].apply(normalize_patient_id)

        return df
