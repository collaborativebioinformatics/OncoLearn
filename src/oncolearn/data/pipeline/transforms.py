"""
Label transform functions for the pipeline DSL.

Moved from ``oncolearn.data.modalities.tabular.parsers.clinical_parser``.
"""
import re
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

_STAGE_PATTERNS = [
    (re.compile(r"stage\s*i(?!v|i)", re.I), 0),   # Stage I  (not II or IV)
    (re.compile(r"stage\s*ii(?!i)", re.I), 1),     # Stage II (not III)
    (re.compile(r"stage\s*iii", re.I), 2),         # Stage III
    (re.compile(r"stage\s*iv", re.I), 3),          # Stage IV
]


def map_ajcc_stage(raw: str) -> Optional[int]:
    """Map a raw AJCC pathologic stage string to an integer class label.

    Regex-based matching is case-insensitive and supports mixed formats:
    XenaBrowser (``"Stage IIB"``), cBioPortal Firehose Legacy (``"Stage IIA"``),
    and cBioPortal Pan-Can Atlas 2018 (``"STAGE IIA"``).

    Returns:
        0 for Stage I, 1 for Stage II, 2 for Stage III, 3 for Stage IV,
        or ``None`` if no pattern matches (e.g. "Stage X", "STAGE X", NaN,
        empty string).
    """
    if not isinstance(raw, str):
        return None
    for pattern, label in _STAGE_PATTERNS:
        if pattern.search(raw):
            return label
    return None


def apply_log2_normalization(df: pd.DataFrame, patient_id_col: str = "patient_id") -> pd.DataFrame:
    """Apply log2(x + 1) normalization in-place to numeric columns of *df*.

    Modifies *df* in-place and returns it.  The *patient_id_col* column is
    excluded from normalization.

    Args:
        df: DataFrame to normalize (mutated in-place).
        patient_id_col: Column to skip during normalization.

    Returns:
        The same *df* object after normalization.
    """
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != patient_id_col]
    df[num_cols] = np.log2(df[num_cols] + 1)
    return df


def make_subtype_transform(class_map: Dict[str, int]) -> Callable[[str], Optional[int]]:
    """Return a label transform function for the given string→int class mapping.

    Args:
        class_map: Dict mapping raw label strings to integer class indices,
                   e.g. ``{"BRCA_LumA": 0, "BRCA_Her2": 1, ...}``.  The actual
                   mapping is defined in the data-specific pipeline config file,
                   not hardcoded in the library — making this reusable for any
                   cancer type or subtype scheme.

    Returns:
        A callable ``(raw: str) -> Optional[int]`` that strips whitespace, looks
        up the value in *class_map*, and returns ``None`` for unrecognised inputs
        (including non-strings and empty strings).  Rows with ``None`` labels are
        dropped by ``TabularDataModule._load_df()``.
    """
    def _transform(raw: object) -> Optional[int]:
        if not isinstance(raw, str):
            return None
        stripped = raw.strip()
        if not stripped:
            return None
        return class_map.get(stripped)

    return _transform
