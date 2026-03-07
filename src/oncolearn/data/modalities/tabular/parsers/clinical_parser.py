"""
Parser for TCGA clinical TSV files from XenaBrowser.

Extends XenabrowserParser with AJCC pathologic stage label extraction and
numeric-column filtering.
"""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .xenabrowser_parser import XenabrowserParser

_STAGE_PATTERNS = [
    (re.compile(r"stage\s*i(?!v|i)", re.I), 0),     # Stage I  (not II or IV)
    (re.compile(r"stage\s*ii(?!i)", re.I), 1),      # Stage II (not III)
    (re.compile(r"stage\s*iii", re.I), 2),          # Stage III
    (re.compile(r"stage\s*iv", re.I), 3),           # Stage IV
]


def _map_stage(raw: str) -> Optional[int]:
    if not isinstance(raw, str):
        return None
    for pattern, label in _STAGE_PATTERNS:
        if pattern.search(raw):
            return label
    return None


class ClinicalParser(XenabrowserParser):
    """
    Parser for TCGA clinical TSV files (e.g. TCGA-BRCA.clinical.tsv).

    On top of the shared XenaBrowser loading logic:
    - Maps ``ajcc_pathologic_stage.diagnoses`` to integer labels via regex;
      rows with Unknown / NaN stage are dropped.
    - Keeps only numeric-coercible feature columns; free text, datetimes,
      and categorical columns are silently discarded.
    """

    STAGE_COL = "ajcc_pathologic_stage.diagnoses"

    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        return super().can_parse(file_path) and "clinical" in file_path.stem.lower()

    @classmethod
    def parse(cls, file_path: Path, stage_col: str = STAGE_COL) -> pd.DataFrame:
        df = cls.load(file_path)

        # Map stage labels; drop rows with no valid stage
        if stage_col in df.columns:
            df["label"] = df[stage_col].apply(_map_stage)
            df = df[df["label"].notna()].copy()
            df["label"] = df["label"].astype(int)
            df = df.drop(columns=[stage_col])

        # Keep only numeric-coercible feature columns
        meta_cols = {"patient_id", "label"}
        feat_candidates = [c for c in df.columns if c not in meta_cols]
        numeric_mask = (
            df[feat_candidates].apply(pd.to_numeric, errors="coerce").notna().any()
        )
        keep_feat_cols = numeric_mask[numeric_mask].index.tolist()
        keep_cols = (
            ["patient_id"]
            + (["label"] if "label" in df.columns else [])
            + keep_feat_cols
        )
        return df[keep_cols]
