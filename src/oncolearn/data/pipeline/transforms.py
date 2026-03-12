"""
Label transform functions for the pipeline DSL.

Moved from ``oncolearn.data.modalities.tabular.parsers.clinical_parser``.
"""
import re
from typing import Optional

_STAGE_PATTERNS = [
    (re.compile(r"stage\s*i(?!v|i)", re.I), 0),   # Stage I  (not II or IV)
    (re.compile(r"stage\s*ii(?!i)", re.I), 1),     # Stage II (not III)
    (re.compile(r"stage\s*iii", re.I), 2),         # Stage III
    (re.compile(r"stage\s*iv", re.I), 3),          # Stage IV
]


def map_ajcc_stage(raw: str) -> Optional[int]:
    """Map a raw AJCC pathologic stage string to an integer class label.

    Regex-based matching supports both XenaBrowser (``"Stage IIB"``) and
    cBioPortal (``"Stage IIA"``) stage strings.

    Returns:
        0 for Stage I, 1 for Stage II, 2 for Stage III, 3 for Stage IV,
        or ``None`` if no pattern matches (e.g. "Stage X", NaN, empty string).
    """
    if not isinstance(raw, str):
        return None
    for pattern, label in _STAGE_PATTERNS:
        if pattern.search(raw):
            return label
    return None
