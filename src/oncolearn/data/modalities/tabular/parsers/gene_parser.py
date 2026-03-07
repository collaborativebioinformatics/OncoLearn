"""
Parser for XenaBrowser gene expression / miRNA TSV files.

Extends XenabrowserParser with PAM50 / subtype label encoding.
"""
from pathlib import Path

import pandas as pd

from .xenabrowser_parser import XenabrowserParser

_SUBTYPE_COLS = ("Subtype", "PAM50", "pam50")


class GeneParser(XenabrowserParser):
    """
    Parser for gene expression and miRNA TSV files from XenaBrowser.

    On top of the shared XenaBrowser loading logic, encodes the PAM50 /
    Subtype column (when present) as an integer ``label`` column, and drops
    rows with missing or ``"Unknown"`` labels.
    """

    @classmethod
    def parse(cls, file_path: Path) -> pd.DataFrame:
        df = cls.load(file_path)

        label_src = next((c for c in _SUBTYPE_COLS if c in df.columns), None)
        if label_src:
            from sklearn.preprocessing import LabelEncoder
            df = df[df[label_src].notna() & (df[label_src] != "Unknown")].copy()
            df["label"] = LabelEncoder().fit_transform(df[label_src])
            df = df.drop(columns=[label_src])

        return df
