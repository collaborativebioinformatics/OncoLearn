"""
cBioPortal dataset classes that download API data to local TSV files.
"""

import csv
from pathlib import Path
from typing import List, Optional

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

        out = Path(output_dir) if output_dir else Path("data/cbioportal") / self.default_subdir
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

        # Determine column order: preserve insertion order, then sort remaining
        all_attrs: list = []
        seen: set = set()
        for row in wide.values():
            for a in row:
                if a not in seen:
                    all_attrs.append(a)
                    seen.add(a)
        all_attrs.sort()

        id_col = "sample"
        rows = sorted(wide.items())

        _write_wide_tsv(dest, id_col, all_attrs, rows)

        if verbose:
            print(f"  Saved {len(rows)} rows × {len(all_attrs)} attributes → {dest}")
        return True

    def _download_molecular(self, client: CBioPortalClient, dest: Path, verbose: bool) -> bool:
        """Fetch molecular data and write gene × sample matrix TSV."""
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for molecular datasets")

        if verbose:
            print(f"  Fetching molecular profile '{self.molecular_profile_id}' "
                  f"(this may take a while for large profiles)…")

        records = client.get_molecular_data(
            molecular_profile_id=self.molecular_profile_id,
            sample_list_id=self.sample_list_id,
            study_id=self.study_id if not self.sample_list_id else None,
            sample_ids=client.get_sample_ids(self.study_id) if not self.sample_list_id else None,
        )

        if not records:
            print(f"  WARNING: No molecular data returned for {self.name}")
            return False

        # Build gene × sample pivot
        # rows = genes, cols = samples (XenaBrowser convention)
        gene_data: dict = {}   # hugoSymbol → {sampleId: value}
        samples_seen: list = []
        samples_set: set = set()

        for rec in records:
            gene_sym = rec.get("gene", {}).get("hugoGeneSymbol") or str(rec.get("entrezGeneId", ""))
            sample = rec["sampleId"]
            value = rec.get("value", "")

            if gene_sym not in gene_data:
                gene_data[gene_sym] = {}
            gene_data[gene_sym][sample] = value

            if sample not in samples_set:
                samples_seen.append(sample)
                samples_set.add(sample)

        samples = sorted(samples_seen)
        gene_col = "Hugo_Symbol"

        with open(dest, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([gene_col] + samples)
            for gene_sym in sorted(gene_data):
                row = [gene_data[gene_sym].get(s, "") for s in samples]
                writer.writerow([gene_sym] + row)

        if verbose:
            print(f"  Saved {len(gene_data)} genes × {len(samples)} samples → {dest}")
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
            sample_ids=client.get_sample_ids(self.study_id) if not self.sample_list_id else None,
        )

        if not records:
            print(f"  WARNING: No mutation data returned for {self.name}")
            return False

        # Flatten nested fields for TSV output
        flat_records = [_flatten_mutation(r) for r in records]
        all_keys: list = []
        keys_set: set = set()
        for rec in flat_records:
            for k in rec:
                if k not in keys_set:
                    all_keys.append(k)
                    keys_set.add(k)

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

        flat_records = [_flatten_sv(r) for r in records]
        all_keys: list = []
        keys_set: set = set()
        for rec in flat_records:
            for k in rec:
                if k not in keys_set:
                    all_keys.append(k)
                    keys_set.add(k)

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
            print(f"  (requires one API call per sample — this may take several minutes)")

        records = client.get_copy_number_segments(study_id=self.study_id)

        if not records:
            print(f"  WARNING: No copy-number segment data returned for {self.name}")
            return False

        _KEEP = {"sampleId", "patientId", "chromosome", "start", "end", "numberOfProbes", "segmentMean"}
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
        """Fetch GENERIC_ASSAY data and write entity × sample matrix TSV."""
        if not self.molecular_profile_id:
            raise ValueError("molecular_profile_id required for generic_assay datasets")

        if verbose:
            print(f"  Fetching generic assay data '{self.molecular_profile_id}'…")

        records = client.get_generic_assay_data(
            molecular_profile_id=self.molecular_profile_id,
            sample_list_id=self.sample_list_id,
            study_id=self.study_id if not self.sample_list_id else None,
            sample_ids=client.get_sample_ids(self.study_id) if not self.sample_list_id else None,
        )

        if not records:
            print(f"  WARNING: No generic assay data returned for {self.name}")
            return False

        # Build entity × sample matrix (rows = stableId, cols = samples)
        entity_data: dict = {}
        samples_seen: list = []
        samples_set: set = set()

        for rec in records:
            entity_id = rec.get("stableId") or rec.get("genericAssayStableId", "")
            sample = rec["sampleId"]
            value = rec.get("value", "")

            if entity_id not in entity_data:
                entity_data[entity_id] = {}
            entity_data[entity_id][sample] = value

            if sample not in samples_set:
                samples_seen.append(sample)
                samples_set.add(sample)

        samples = sorted(samples_seen)

        with open(dest, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["stableId"] + samples)
            for entity_id in sorted(entity_data):
                row = [entity_data[entity_id].get(s, "") for s in samples]
                writer.writerow([entity_id] + row)

        if verbose:
            print(f"  Saved {len(entity_data)} entities × {len(samples)} samples → {dest}")
        return True


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _write_wide_tsv(dest: Path, id_col: str, attrs: list, rows: list) -> None:
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([id_col] + attrs)
        for row_id, row_data in rows:
            writer.writerow([row_id] + [row_data.get(a, "") for a in attrs])


def _flatten_sv(rec: dict) -> dict:
    """Flatten a cBioPortal structural variant record into a single-level dict."""
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


def _flatten_mutation(rec: dict) -> dict:
    """Flatten a cBioPortal mutation record into a single-level dict."""
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
