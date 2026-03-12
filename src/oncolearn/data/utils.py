"""Shared utilities for the oncolearn.data package."""


def normalize_patient_id(pid: str) -> str:
    """Truncate a TCGA patient/sample ID to the standard 12-character format.

    TCGA sample IDs (15-char, e.g. ``"TCGA-A7-A0CG-01"``) are shortened to
    patient IDs (12-char, e.g. ``"TCGA-A7-A0CG"``).  Non-TCGA IDs are
    returned unchanged.
    """
    if isinstance(pid, str) and pid.upper().startswith("TCGA-"):
        return pid[:12]
    return pid
