"""
Reader for XenaBrowser TSV data files.

Thin adapter over :class:`XenabrowserParser` that implements the
:class:`BaseReader` protocol.
"""
from pathlib import Path

import pandas as pd

from .base import BaseReader
from oncolearn.data.modalities.loaders.tabular_loader import XenabrowserParser


class XenabrowserReader(BaseReader):
    """Read XenaBrowser TSV files from a base directory.

    Dataset names are treated as filenames relative to *base_dir*.

    Args:
        config_path: Unused for XenaBrowser (kept for API symmetry with
                     :class:`CbioPortalReader`).  Pass ``""`` or a label string.
        base_dir: Directory containing the TSV files
                  (e.g. ``"data/sources/xenabrowser/TCGA-BRCA"``).
    """

    def __init__(self, config_path: str, base_dir: str) -> None:
        self._base_dir = Path(base_dir)

    def read(self, name: str) -> pd.DataFrame:
        """Load a TSV file from the base directory.

        Args:
            name: Filename relative to *base_dir*
                  (e.g. ``"TCGA-BRCA.mirna.tsv"``).

        Returns:
            DataFrame with a ``patient_id`` column normalized by
            :meth:`XenabrowserParser.load`.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the file cannot be parsed.
        """
        path = self._base_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"XenaBrowser data file not found: {path}"
            )
        return XenabrowserParser.load(path)
