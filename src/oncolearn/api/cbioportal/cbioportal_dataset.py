"""
cBioPortal dataset classes that download API data to local TSV files.
"""

import csv
import math
from pathlib import Path
from typing import Callable, List, Optional

from tqdm import tqdm

from ..dataset import DataCategory, Dataset
from .client import CBioPortalClient


class CBioPortalDataset(Dataset):
    """
    A single downloadable dataset from cBioPortal.

    Supports two dataset types:

    * ``"clinical"`` – fetches clinical attributes and writes a wide-format TSV
      with one row per patient/sample and one column per attribute.
    * ``"molecular"`` – fetches a molecular profile (expression, CNV, …) and
      writes a gene × sample matrix TSV matching the XenaBrowser convention.
    * ``"mutations"`` – fetches MAF records and writes a long-format TSV.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: DataCategory,
        study_id: str,
        filename: str,
        default_subdir: str,
        dataset_type: str,                          # "clinical" | "molecular" | "mutations"
        # clinical-specific
        clinical_data_type: str = "PATIENT",        # "PATIENT" or "SAMPLE"
        attribute_ids: Optional[List[str]] = None,
        # molecular / mutations-specific
        molecular_profile_id: Optional[str] = None,
        sample_list_id: Optional[str] = None,
        batch_size: int = 200,
        # shared
        base_url: str = "https://www.cbioportal.org/api",
    ):
        super().__init__(name=name, description=description)
        self.DATA_CATEGORY = category
        self.study_id = study_id
        self.filename = filename
        self.default_subdir = default_subdir
        self.dataset_type = dataset_type
        self.clinical_data_type = clinical_data_type
        self.attribute_ids = attribute_ids or []
        self.molecular_profile_id = molecular_profile_id
        self.sample_list_id = sample_list_id
        self.batch_size = batch_size
        self.base_url = base_url

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def download(
        self,
        output_dir: Optional[str] = None,
        confirm: bool = True,
        verbose: bool = True,
    ) -> bool:
        """
        Fetch data from cBioPortal and write it to *output_dir* as a TSV.

        Returns ``True`` on success.
        """
        from oncolearn.cli.utils.download import ensure_directory

        out = Path(output_dir) if output_dir else Path("data/sources/cbioportal") / self.default_subdir
        ensure_directory(out)
        dest = out / self.filename

        if dest.exists():
            if verbose:
                print(f"  Skipping {self.filename} (already exists)")
            return True

        client = CBioPortalClient(base_url=self.base_url)

        try:
            if self.dataset_type == "clinical":
                return self._download_clinical(client, dest, verbose)
            elif self.dataset_type == "molecular":
                return self._download_molecular(client, dest, verbose)
            elif self.dataset_type == "mutations":
                return self._download_mutations(client, dest, verbose)
            elif self.dataset_type == "structural_variants":
                return self._download_structural_variants(client, dest, verbose)
            elif self.dataset_type == "generic_assay":
                return self._download_generic_assay(client, dest, verbose)
            elif self.dataset_type == "copy_number_segments":
                return self._download_copy_number_segments(client, dest, verbose)
            else:
                raise ValueError(f"Unknown dataset_type: {self.dataset_type!r}")
        except Exception as exc:
            print(f"  ERROR downloading {self.name}: {exc}")
            return False

    # ------------------------------------------------------------------
    #  Internal download methods
    # ------------------------------------------------------------------

    def _resolve_sample_ids(self, client: CBioPortalClient) -> List[str]:
        """Resolve sample IDs from sample_list_id if set, otherwise from study_id."""
        if self.sample_list_id:
            return client.get_sample_list_ids(self.sample_list_id)
        return client.get_sample_ids(self.study_id)

    def _download_clinical(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """Fetch clinical data and write wide-format TSV."""
        if verbose:
            attr_desc = f"attributes {self.attribute_ids}" if self.attribute_ids else "all attributes"
            print(f"  Fetching {self.clinical_data_type.lower()} clinical data "
                  f"({attr_desc}) for study '{self.study_id}'…")

        records = client.get_clinical_data(
            self.study_id,
            clinical_data_type=self.clinical_data_type,
            attribute_ids=self.attribute_ids if self.attribute_ids else None,
        )

        if not records:
            print(f"  WARNING: No clinical data returned for {self.name}")
            return False

        # Pivot long → wide
        # Key is patientId for PATIENT-level, sampleId for SAMPLE-level
        id_key = "sampleId" if self.clinical_data_type == "SAMPLE" else "patientId"
        wide: dict = {}
        for rec in records:
            row_id = rec.get(id_key, "")
            attr = rec["clinicalAttributeId"]
            val = rec.get("value", "")
            if row_id not in wide:
                wide[row_id] = {}
            wide[row_id][attr] = val

        all_attrs = sorted({a for row in wide.values() for a in row})

        id_col = "sample"
        rows = sorted(wide.items())

        _write_wide_tsv(dest, id_col, all_attrs, rows)

        if verbose:
            print(f"  Saved {len(rows)} rows × {len(all_attrs)} attributes → {dest}")
        return True

    def _download_molecular(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """
        Fetch molecular data in sample batches and write a sample × gene TSV.

        Rows are samples, columns are genes/probes (sample-major format).  This
        avoids loading the full gene × sample matrix into memory at once, which
        is prohibitive for large profiles such as HM450 methylation
        (~485 K probes × 800+ samples ≈ 388 M records).
        """
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for molecular datasets")

        sample_ids = self._resolve_sample_ids(client)
        if not sample_ids:
            print(f"  WARNING: No samples found for {self.name}")
            return False

        n_batches = math.ceil(len(sample_ids) / self.batch_size)
        if verbose:
            print(f"  Fetching molecular profile '{self.molecular_profile_id}' "
                  f"({len(sample_ids)} samples, {n_batches} batches of {self.batch_size})…")

        def _gene_key(rec: dict) -> str:
            return (rec.get("gene") or {}).get("hugoGeneSymbol") \
                   or str(rec.get("entrezGeneId", ""))

        batches = client.get_molecular_data_batched(
            self.molecular_profile_id, sample_ids, batch_size=self.batch_size
        )
        total = self._write_batched_sample_matrix(
            batches, _gene_key, dest, n_batches, self.molecular_profile_id, verbose
        )
        if total == 0:
            print(f"  WARNING: No molecular data returned for {self.name}")
            return False

        if verbose:
            print(f"  Saved {total} samples → {dest}")
        return True

    def _download_mutations(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """Fetch mutation records and write long-format TSV."""
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for mutation datasets")

        if verbose:
            print(f"  Fetching mutations from profile '{self.molecular_profile_id}'…")

        records = client.get_mutations(
            molecular_profile_id=self.molecular_profile_id,
            sample_list_id=self.sample_list_id,
            study_id=self.study_id if not self.sample_list_id else None,
            sample_ids=self._resolve_sample_ids(client) if not self.sample_list_id else None,
        )

        if not records:
            print(f"  WARNING: No mutation data returned for {self.name}")
            return False

        flat_records = [_flatten_record(r) for r in records]
        all_keys = _ordered_keys(flat_records)

        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t",
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(flat_records)

        if verbose:
            print(f"  Saved {len(flat_records)} mutation records → {dest}")
        return True

    def _download_structural_variants(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """Fetch structural variant records and write long-format TSV."""
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for structural_variants datasets")

        if verbose:
            print(f"  Fetching structural variants from profile '{self.molecular_profile_id}'…")

        records = client.get_structural_variants(
            molecular_profile_id=self.molecular_profile_id,
            study_id=self.study_id,
        )

        if not records:
            print(f"  WARNING: No structural variant data returned for {self.name}")
            return False

        flat_records = [_flatten_record(r) for r in records]
        all_keys = _ordered_keys(flat_records)

        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t",
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(flat_records)

        if verbose:
            print(f"  Saved {len(flat_records)} structural variant records → {dest}")
        return True

    def _download_copy_number_segments(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """Fetch CN segments (one GET per sample) and write long-format TSV."""
        if verbose:
            print(f"  Fetching copy-number segments for study '{self.study_id}'…")
            print("  (requires one API call per sample — this may take several minutes)")

        records = client.get_copy_number_segments(study_id=self.study_id, show_progress=verbose)

        if not records:
            print(f"  WARNING: No copy-number segment data returned for {self.name}")
            return False

        fieldnames = ["sampleId", "patientId", "chromosome", "start", "end", "numberOfProbes", "segmentMean"]

        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(records)

        n_samples = len({r["sampleId"] for r in records})
        if verbose:
            print(f"  Saved {len(records)} segments across {n_samples} samples → {dest}")
        return True

    def _download_generic_assay(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """
        Fetch GENERIC_ASSAY data in sample batches and write a sample × entity TSV.

        Rows are samples, columns are assay entity stable IDs (sample-major format).
        """
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for generic_assay datasets")

        sample_ids = self._resolve_sample_ids(client)
        if not sample_ids:
            print(f"  WARNING: No samples found for {self.name}")
            return False

        n_batches = math.ceil(len(sample_ids) / self.batch_size)
        if verbose:
            print(f"  Fetching generic assay data '{self.molecular_profile_id}' "
                  f"({len(sample_ids)} samples, {n_batches} batches of {self.batch_size})…")

        def _entity_key(rec: dict) -> str:
            return rec.get("stableId") or rec.get("genericAssayStableId", "")

        batches = client.get_generic_assay_data_batched(
            self.molecular_profile_id, sample_ids, batch_size=self.batch_size
        )
        total = self._write_batched_sample_matrix(
            batches, _entity_key, dest, n_batches, self.molecular_profile_id, verbose
        )
        if total == 0:
            print(f"  WARNING: No generic assay data returned for {self.name}")
            return False

        if verbose:
            print(f"  Saved {total} samples → {dest}")
        return True

    def _write_batched_sample_matrix(
        self,
        batch_iter,
        key_extractor: Callable[[dict], str],
        dest: Path,
        n_batches: int,
        desc: str,
        verbose: bool,
    ) -> int:
        """
        Stream batches of records into a sample × feature TSV.

        Assumes all samples share the same feature set (true for dense cBioPortal
        profiles). Column order is determined from the first non-empty batch.

        Returns the number of samples written, or 0 if no data was received.
        """
        cols_ordered: Optional[List[str]] = None
        total_samples = 0

        with open(dest, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            with tqdm(batch_iter, total=n_batches, unit="batch",
                      desc=desc, disable=not verbose, leave=False) as pbar:
                for batch_records in pbar:
                    sample_data: dict = {}
                    for rec in batch_records:
                        key = key_extractor(rec)
                        sample = rec["sampleId"]
                        if sample not in sample_data:
                            sample_data[sample] = {}
                        sample_data[sample][key] = rec.get("value", "")

                    if not sample_data:
                        continue

                    if cols_ordered is None:
                        # All samples in a profile share the same gene/probe set;
                        # read column order from the first sample (O(n_cols) not O(batch×n_cols)).
                        cols_ordered = sorted(next(iter(sample_data.values())))
                        writer.writerow(["sample"] + cols_ordered)

                    for sid, vals in sample_data.items():
                        writer.writerow([sid] + [vals.get(c, "") for c in cols_ordered])
                        total_samples += 1
                    pbar.set_postfix(samples=total_samples)

        if not cols_ordered:
            dest.unlink(missing_ok=True)
            return 0

        return total_samples


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _write_wide_tsv(dest: Path, id_col: str, attrs: list, rows: list) -> None:
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([id_col] + attrs)
        for row_id, row_data in rows:
            writer.writerow([row_id] + [row_data.get(a, "") for a in attrs])


def _flatten_record(rec: dict) -> dict:
    """Flatten a cBioPortal record with nested dicts/lists into a single-level dict."""
    flat = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = sub_v
        elif isinstance(v, list):
            flat[k] = ";".join(str(i) for i in v)
        else:
            flat[k] = v
    return flat


def _ordered_keys(records: List[dict]) -> List[str]:
    """Collect all keys from a list of dicts, preserving first-seen insertion order."""
    keys: list = []
    seen: set = set()
    for rec in records:
        for k in rec:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    return keys
