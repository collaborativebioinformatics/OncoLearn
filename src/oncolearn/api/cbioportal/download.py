"""
High-level download functions for cBioPortal data.

Mirrors the interface of oncolearn.api.xenabrowser.download and
oncolearn.api.tcia.download so they can be called uniformly from the CLI.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .builder import CBioPortalCohortBuilder
from .client import CBioPortalClient


def get_available_cohorts() -> List[str]:
    """Return cohort codes with YAML configs in data/cbioportal/configs/."""
    return CBioPortalCohortBuilder().list_available_cohorts()


def list_studies(
    keyword: Optional[str] = None,
    cancer_type_id: Optional[str] = None,
    base_url: str = "https://www.cbioportal.org/api",
) -> List[Dict]:
    """
    Query cBioPortal directly for available studies.

    Unlike the config-file-based :func:`get_available_cohorts`, this hits
    the live API and returns all studies matching the filters.

    Args:
        keyword: Free-text search (matched against study name/description).
        cancer_type_id: e.g. ``"brca"`` to filter breast cancer studies.
        base_url: Override the cBioPortal base URL.

    Returns:
        List of study dicts (keys: studyId, name, cancerTypeId, …).
    """
    client = CBioPortalClient(base_url=base_url)
    return client.list_studies(keyword=keyword, cancer_type_id=cancer_type_id)


def download_cohort(
    cohort_code: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    confirm: bool = True,
    config_dir: Optional[Path] = None,
) -> bool:
    """
    Download a single cohort by its config-file code (e.g. ``"BRCA"``).

    Returns ``True`` on success.
    """
    try:
        builder = CBioPortalCohortBuilder(config_dir=config_dir)
        cohort = builder.build_cohort(cohort_code)
        cohort.download(output_dir=output_dir, verbose=verbose, confirm=confirm)
        return True
    except Exception as exc:
        print(f"ERROR downloading cBioPortal cohort '{cohort_code}': {exc}")
        return False


def download_cohorts(
    cohort_codes: List[str],
    output_dir: Optional[str] = None,
    verbose: bool = True,
    confirm: bool = True,
    config_dir: Optional[Path] = None,
) -> Dict[str, bool]:
    """Download multiple cohorts. Returns ``{code: success}`` mapping."""
    results = {}
    for code in cohort_codes:
        results[code] = download_cohort(
            code,
            output_dir=output_dir,
            verbose=verbose,
            confirm=confirm,
            config_dir=config_dir,
        )
    return results


def download_all(
    output_dir: Optional[str] = None,
    verbose: bool = True,
    confirm: bool = True,
    config_dir: Optional[Path] = None,
) -> Dict[str, bool]:
    """Download all cohorts that have a YAML config file."""
    cohorts = get_available_cohorts()
    if not cohorts:
        print("No cBioPortal cohort configs found in data/cbioportal/configs/")
        return {}
    return download_cohorts(
        cohorts,
        output_dir=output_dir,
        verbose=verbose,
        confirm=confirm,
        config_dir=config_dir,
    )
