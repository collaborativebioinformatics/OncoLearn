"""
Thin REST client for the cBioPortal public API (v3).

Docs: https://www.cbioportal.org/api/swagger-ui
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


_DEFAULT_BASE_URL = "https://www.cbioportal.org/api"
_PAGE_SIZE = 10_000_000  # cBioPortal returns all data when size is very large


class CBioPortalAPIError(Exception):
    """Raised when the cBioPortal API returns a non-2xx response."""


class CBioPortalClient:
    """
    Minimal HTTP client wrapping the cBioPortal REST API.

    All methods return parsed JSON (list or dict). Pagination is handled
    internally; callers receive the full result set.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise CBioPortalAPIError(
                f"HTTP {exc.code} for GET {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CBioPortalAPIError(f"Network error for GET {url}: {exc.reason}") from exc

    def _post(self, path: str, body: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")[:500]
            raise CBioPortalAPIError(
                f"HTTP {exc.code} for POST {url}: {exc.reason} — {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CBioPortalAPIError(f"Network error for POST {url}: {exc.reason}") from exc

    # ------------------------------------------------------------------ #
    #  Studies                                                             #
    # ------------------------------------------------------------------ #

    def list_studies(
        self,
        keyword: Optional[str] = None,
        cancer_type_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return all studies, optionally filtered by keyword or cancer type."""
        params: Dict[str, Any] = {"pageSize": _PAGE_SIZE, "pageNumber": 0}
        if keyword:
            params["keyword"] = keyword
        results = self._get("/studies", params)
        if cancer_type_id:
            results = [s for s in results if s.get("cancerTypeId") == cancer_type_id]
        return results

    def get_study(self, study_id: str) -> Dict:
        """Return metadata for a single study."""
        return self._get(f"/studies/{study_id}")

    # ------------------------------------------------------------------ #
    #  Samples                                                             #
    # ------------------------------------------------------------------ #

    def get_samples(self, study_id: str) -> List[Dict]:
        """Return all samples for a study."""
        return self._get(
            f"/studies/{study_id}/samples",
            {"pageSize": _PAGE_SIZE, "pageNumber": 0},
        )

    def get_sample_ids(self, study_id: str) -> List[str]:
        """Return a flat list of sample IDs."""
        return [s["sampleId"] for s in self.get_samples(study_id)]

    # ------------------------------------------------------------------ #
    #  Clinical data                                                       #
    # ------------------------------------------------------------------ #

    def get_clinical_attributes(self, study_id: str) -> List[Dict]:
        """Return all clinical attribute definitions for a study."""
        return self._get(f"/studies/{study_id}/clinical-attributes")

    def get_clinical_data(
        self,
        study_id: str,
        clinical_data_type: str = "PATIENT",
        attribute_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Return clinical data records in long format.

        Args:
            study_id: e.g. ``"brca_tcga"``
            clinical_data_type: ``"PATIENT"`` or ``"SAMPLE"``
            attribute_ids: Subset of attribute IDs to fetch; ``None`` = all.

        Returns:
            List of dicts with keys ``patientId``, ``sampleId``,
            ``clinicalAttributeId``, ``value``.
        """
        params: Dict[str, Any] = {
            "clinicalDataType": clinical_data_type,
            "pageSize": _PAGE_SIZE,
            "pageNumber": 0,
        }
        records = self._get(f"/studies/{study_id}/clinical-data", params)
        if attribute_ids:
            attr_set = set(attribute_ids)
            records = [r for r in records if r["clinicalAttributeId"] in attr_set]
        return records

    # ------------------------------------------------------------------ #
    #  Molecular profiles & data                                           #
    # ------------------------------------------------------------------ #

    def get_molecular_profiles(self, study_id: str) -> List[Dict]:
        """Return all molecular profile definitions for a study."""
        return self._get(f"/studies/{study_id}/molecular-profiles")

    def get_molecular_data(
        self,
        molecular_profile_id: str,
        sample_list_id: Optional[str] = None,
        sample_ids: Optional[List[str]] = None,
        study_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch molecular data for a profile.

        Exactly one of ``sample_list_id`` or ``sample_ids`` must be provided.
        When ``sample_ids`` is given, ``study_id`` is also required.

        Returns records with keys ``sampleId``, ``entrezGeneId``, ``value``,
        and nested ``gene`` dict (``hugoGeneSymbol``).
        """
        if sample_list_id:
            body = {"sampleListId": sample_list_id}
        elif sample_ids and study_id:
            body = {
                "sampleMolecularIdentifiers": [
                    {"sampleId": sid, "molecularProfileId": molecular_profile_id}
                    for sid in sample_ids
                ]
            }
        else:
            raise ValueError("Provide sample_list_id or both sample_ids + study_id")

        return self._post(
            f"/molecular-profiles/{molecular_profile_id}/molecular-data/fetch",
            body,
        )

    def get_mutations(
        self,
        molecular_profile_id: str,
        sample_list_id: Optional[str] = None,
        sample_ids: Optional[List[str]] = None,
        study_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch mutation (MAF) records for a mutation profile.

        Same arguments as :meth:`get_molecular_data`.
        """
        if sample_list_id:
            body = {"sampleListId": sample_list_id}
        elif sample_ids and study_id:
            body = {
                "sampleMolecularIdentifiers": [
                    {"sampleId": sid, "molecularProfileId": molecular_profile_id}
                    for sid in sample_ids
                ]
            }
        else:
            raise ValueError("Provide sample_list_id or both sample_ids + study_id")

        return self._post(
            f"/molecular-profiles/{molecular_profile_id}/mutations/fetch",
            body,
            {"projection": "DETAILED"},
        )

    def get_sample_lists(self, study_id: str) -> List[Dict]:
        """Return all sample lists (e.g. 'all', 'rna_seq_v2_mrna') for a study."""
        return self._get(f"/studies/{study_id}/sample-lists")
