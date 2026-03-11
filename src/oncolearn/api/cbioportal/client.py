"""
Thin REST client for the cBioPortal public API (v3).

Docs: https://www.cbioportal.org/api/swagger-ui
"""

import time
from typing import Any, Dict, Generator, List, Optional

import requests
from tqdm import tqdm


_DEFAULT_BASE_URL = "https://www.cbioportal.org/api"
_PAGE_SIZE = 10_000_000  # cBioPortal returns all data when size is very large
_USER_AGENT = "OncoLearn/1.0 (https://github.com/oncolearn; research use)"
# Retry on these transient HTTP status codes
_RETRYABLE = {429, 500, 502, 503, 504}
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 2.0   # seconds; doubles each attempt (2 → 4 → 8)


class CBioPortalAPIError(Exception):
    """Raised when the cBioPortal API returns a non-2xx response."""


class CBioPortalClient:
    """
    Minimal HTTP client wrapping the cBioPortal REST API.

    Uses a persistent ``requests.Session`` for connection reuse (keep-alive)
    and automatic gzip decompression — both significantly reduce transfer time
    for the large molecular-profile payloads.

    All methods return parsed JSON (list or dict). Pagination is handled
    internally; callers receive the full result set.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        })

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return self._send("GET", url, params=params)

    def _post(self, path: str, body: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return self._send("POST", url, params=params, json=body)

    def _send(self, method: str, url: str, **kwargs) -> Any:
        """Execute a request with exponential-backoff retry on transient errors.

        Respects the ``Retry-After`` response header on 429 responses so we
        wait exactly as long as the server requests rather than guessing.
        """
        delay = _RETRY_BACKOFF
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if resp.status_code in _RETRYABLE and attempt < _RETRY_ATTEMPTS:
                    wait = float(resp.headers.get("Retry-After", delay))
                    time.sleep(wait)
                    delay *= 2
                    continue
                if not resp.ok:
                    body_text = resp.text[:500] if method == "POST" else ""
                    suffix = f" — {body_text}" if body_text else ""
                    raise CBioPortalAPIError(
                        f"HTTP {resp.status_code} for {method} {url}: "
                        f"{resp.reason}{suffix}"
                    )
                return resp.json()
            except requests.exceptions.ConnectionError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CBioPortalAPIError(
                    f"Network error for {method} {url}: {exc}"
                ) from exc
            except requests.exceptions.Timeout as exc:
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CBioPortalAPIError(
                    f"Timeout for {method} {url}"
                ) from exc

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
            body: Dict[str, Any] = {"sampleListId": sample_list_id}
        elif sample_ids:
            body = {"sampleIds": sample_ids}
        else:
            raise ValueError("Provide sample_list_id or sample_ids")

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
            body: Dict[str, Any] = {"sampleListId": sample_list_id}
        elif sample_ids:
            body = {"sampleIds": sample_ids}
        else:
            raise ValueError("Provide sample_list_id or sample_ids")

        return self._post(
            f"/molecular-profiles/{molecular_profile_id}/mutations/fetch",
            body,
            {"projection": "DETAILED"},
        )

    def get_sample_lists(self, study_id: str) -> List[Dict]:
        """Return all sample lists (e.g. 'all', 'rna_seq_v2_mrna') for a study."""
        return self._get(f"/studies/{study_id}/sample-lists")

    def get_sample_list_ids(self, sample_list_id: str) -> List[str]:
        """Return sample IDs belonging to a specific sample list."""
        return self._get(f"/sample-lists/{sample_list_id}/sample-ids")

    def get_molecular_data_batched(
        self,
        molecular_profile_id: str,
        sample_ids: List[str],
        batch_size: int = 200,
    ) -> Generator[List[Dict], None, None]:
        """
        Yield molecular data records for *sample_ids* in batches of *batch_size*.

        Uses ``{"sampleIds": [...]}`` — the correct body format for the
        single-study ``/molecular-profiles/{id}/molecular-data/fetch`` endpoint.
        (``sampleMolecularIdentifiers`` is for the multi-study endpoint only.)

        Batches are yielded in the same order as *sample_ids* so that the
        output file row order is deterministic and reproducible.
        """
        for i in range(0, len(sample_ids), batch_size):
            yield self._post(
                f"/molecular-profiles/{molecular_profile_id}/molecular-data/fetch",
                {"sampleIds": sample_ids[i:i + batch_size]},
            )

    def get_generic_assay_data_batched(
        self,
        molecular_profile_id: str,
        sample_ids: List[str],
        batch_size: int = 200,
    ) -> Generator[List[Dict], None, None]:
        """
        Yield GENERIC_ASSAY records for *sample_ids* in batches of *batch_size*.

        Each yielded value is the raw list of records (keys: ``stableId``,
        ``sampleId``, ``value``).  Batches yielded in *sample_ids* order.
        """
        for i in range(0, len(sample_ids), batch_size):
            yield self._post(
                f"/generic_assay_data/{molecular_profile_id}/fetch",
                {"sampleIds": sample_ids[i:i + batch_size]},
            )

    def get_copy_number_segments(
        self,
        study_id: str,
        sample_ids: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> List[Dict]:
        """
        Fetch all copy-number segments for a study using concurrent per-sample GETs.

        The cBioPortal batch POST endpoint (``/copy-number-segments/fetch``) is
        non-functional for these studies; the only working path is the per-sample
        GET endpoint.  ``pageSize`` is capped at 10 000 server-side.

        *max_workers* requests are issued concurrently (default 10), giving
        roughly a 10× speedup over sequential fetching.

        Returns:
            List of dicts with keys ``sampleId``, ``patientId``, ``chromosome``,
            ``start``, ``end``, ``numberOfProbes``, ``segmentMean``.
        """
        _SEGMENT_PAGE_SIZE = 10_000
        if sample_ids is None:
            sample_ids = self.get_sample_ids(study_id)

        all_segments: List[Dict] = []
        with tqdm(sample_ids, unit="sample", desc=f"{study_id} CN segments",
                  disable=not show_progress, leave=False) as pbar:
            for sid in pbar:
                all_segments.extend(self._get(
                    f"/studies/{study_id}/samples/{sid}/copy-number-segments",
                    {"pageSize": _SEGMENT_PAGE_SIZE, "pageNumber": 0},
                ))
                pbar.set_postfix(segments=len(all_segments))
        return all_segments

    def get_structural_variants(
        self,
        molecular_profile_id: str,
        sample_ids: Optional[List[str]] = None,
        study_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch structural variant records for a SV profile.

        Unlike mutations/molecular data, the structural variant endpoint does not
        accept a ``sampleListId``; sample identifiers must be passed explicitly.
        """
        if not sample_ids:
            if not study_id:
                raise ValueError("study_id required when sample_ids not provided")
            sample_ids = self.get_sample_ids(study_id)

        body = {
            "sampleMolecularIdentifiers": [
                {"sampleId": sid, "molecularProfileId": molecular_profile_id}
                for sid in sample_ids
            ],
            "entrezGeneIds": [],
        }
        return self._post(
            f"/molecular-profiles/{molecular_profile_id}/structural-variant/fetch",
            body,
        )

    def get_generic_assay_data(
        self,
        molecular_profile_id: str,
        sample_list_id: Optional[str] = None,
        sample_ids: Optional[List[str]] = None,
        study_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch GENERIC_ASSAY data (phosphoproteomics, methylation probes,
        arm-level CNA, genetic ancestry, etc.).

        Returns records with keys ``stableId``, ``sampleId``, ``value``.
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
            f"/generic_assay_data/{molecular_profile_id}/fetch",
            body,
        )
