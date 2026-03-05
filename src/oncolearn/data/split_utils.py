"""
Utilities for reading and writing patient-ID split files.

Split files contain one patient_id per line, e.g.::

    TCGA-A1-A0SD
    TCGA-A2-A0CM
    ...
"""

from pathlib import Path
from typing import Iterable, Optional, Set


def read_id_file(path: Path) -> Optional[Set[str]]:
    """Read one patient_id per line from *path*.

    Returns:
        Set of patient IDs, or ``None`` if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    ids = set()
    with path.open("r") as f:
        for line in f:
            pid = line.strip()
            if pid:
                ids.add(pid)
    return ids


def write_id_file(path: Path, ids: Iterable[str]) -> None:
    """Write patient IDs one per line to *path*, sorted for reproducibility.

    Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")
