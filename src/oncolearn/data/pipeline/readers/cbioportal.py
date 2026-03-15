"""
Reader for cBioPortal TSV data files.

Parses the cBioPortal cohort config YAML to build a ``{name: filename}``
lookup, then reads the corresponding TSV files from disk.
"""
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from .base import BaseReader
from oncolearn.data.utils import normalize_patient_id

# Candidate column names for the patient/sample ID in cBioPortal files.
_ID_COLUMN_CANDIDATES = ("sample", "PATIENT_ID", "Patient ID", "patientId")


class CbioPortalReader(BaseReader):
    """Read cBioPortal TSV files referenced by a cohort config YAML.

    The cohort config (e.g. ``data/configs/sources/cbioportal/brca_tcga.yaml``) maps
    dataset names to filenames and provides the ``default_output_subdir`` that
    locates the downloaded files under *base_dir*.

    Downloaded files are expected at::

        <base_dir>/<default_output_subdir>/<filename>

    Args:
        config_path: Path to the cBioPortal cohort config YAML.
        base_dir: Root directory where cBioPortal data is stored
                  (e.g. ``"data/sources/cbioportal"``).
    """

    def __init__(self, config_path: str, base_dir: str) -> None:
        self._config_path = config_path
        self._base_dir = Path(base_dir)
        self._subdir: str = ""
        self._datasets: Dict[str, dict] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazily parse the cohort config YAML on first read."""
        if self._loaded:
            return
        with open(self._config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self._subdir = cfg.get("cohort", {}).get("default_output_subdir", "")
        self._datasets = {ds["name"]: ds for ds in cfg.get("datasets", [])}
        self._loaded = True

    def read(self, name: str) -> pd.DataFrame:
        """Load a named dataset from the cBioPortal download directory.

        Args:
            name: Dataset name as defined in the cohort config YAML
                  (e.g. ``"clinical_patient"``, ``"rna_seq_v2_mrna"``).

        Returns:
            DataFrame with a ``patient_id`` column (12-char TCGA format) and
            all data columns.  Duplicate patient rows are dropped.

        Raises:
            KeyError: If *name* is not in the cohort config.
            FileNotFoundError: If the TSV file does not exist on disk.
        """
        self._ensure_loaded()
        ds_meta = self._datasets.get(name)
        if ds_meta is None:
            raise KeyError(
                f"Dataset '{name}' not found in cBioPortal config. "
                f"Available: {sorted(self._datasets)}"
            )

        filename = ds_meta["filename"]
        path = self._base_dir / self._subdir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"cBioPortal data file not found: {path}\n"
                f"Run the cBioPortal download CLI to fetch the data first."
            )

        df = pd.read_csv(str(path), sep="\t", low_memory=False)

        # Normalize the patient/sample ID column to "patient_id"
        for candidate in _ID_COLUMN_CANDIDATES:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "patient_id"})
                break

        # Truncate to 12-char TCGA patient ID (sample IDs are 15-char TCGA-XX-XXXX-XXX)
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].apply(normalize_patient_id)
            df = df.drop_duplicates(subset=["patient_id"])

        return df
