"""
Cohort builder that constructs cBioPortal cohorts from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..cohort import Cohort
from ..cohort_builder import CohortBuilder as BaseCohortBuilder
from ..dataset import DataCategory
from .cbioportal_dataset import CBioPortalDataset


class CBioPortalCohortBuilder(BaseCohortBuilder):
    """
    Builder class that constructs Cohort objects from YAML config files.

    Config files live in ``data/cbioportal/configs/`` by default and follow
    the same two-section pattern as the Xena/TCIA builders::

        cohort:
          name: TCGA Breast Invasive Carcinoma
          code: BRCA
          study_id: brca_tcga
          description: "..."
          default_output_subdir: TCGA-BRCA

        datasets:
          - name: clinical
            description: "Patient clinical attributes"
            category: clinical
            type: clinical
            clinical_data_type: PATIENT   # PATIENT | SAMPLE
            attribute_ids: []             # empty = all
            filename: TCGA-BRCA.clinical.tsv
          - name: pam50
            ...
    """

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_dir = project_root / "data" / "cbioportal" / "configs"
        super().__init__(config_dir)
        self.config_dir = Path(config_dir)

    # ------------------------------------------------------------------
    #  Category parsing
    # ------------------------------------------------------------------

    def _parse_category(self, category_str: str) -> DataCategory:
        category_map = {
            "image":            DataCategory.IMAGE,
            "clinical":         DataCategory.CLINICAL,
            "phenotype":        DataCategory.CLINICAL,
            "mrna_seq":         DataCategory.MRNA_SEQ,
            "mrna":             DataCategory.MRNA_SEQ,
            "rna_seq":          DataCategory.MRNA_SEQ,
            "dna_seq":          DataCategory.DNA_SEQ,
            "dna":              DataCategory.DNA_SEQ,
            "mirna_seq":        DataCategory.MIRNA_SEQ,
            "mirna":            DataCategory.MIRNA_SEQ,
            "protein":          DataCategory.PROTEIN,
            "methylation":      DataCategory.METHYLATION,
            "cnv":              DataCategory.CNV,
            "copy_number":      DataCategory.CNV,
            "mutation":         DataCategory.MUTATION,
            "mutations":        DataCategory.MUTATION,
            "snp":              DataCategory.SNP,
            "transcriptome":    DataCategory.TRANSCRIPTOME,
            "metabolomics":     DataCategory.METABOLOMICS,
            "proteomics":       DataCategory.PROTEOMICS,
            "genomics":         DataCategory.GENOMICS,
            "manifest":         DataCategory.MANIFEST,
            "multimodal":       DataCategory.MULTIMODAL,
        }
        return category_map.get(category_str.lower(), DataCategory.CLINICAL)

    # ------------------------------------------------------------------
    #  Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(
        self, dataset_config: Dict[str, Any], cohort_info: Dict[str, Any]
    ) -> CBioPortalDataset:
        category = self._parse_category(dataset_config.get("category", "clinical"))
        dataset_type = dataset_config.get("type", "clinical")

        # Dataset-level study_id overrides the cohort-level one (e.g. for PAM50)
        study_id = dataset_config.get("study_id") or cohort_info["study_id"]

        return CBioPortalDataset(
            name=dataset_config["name"],
            description=dataset_config.get("description", ""),
            category=category,
            study_id=study_id,
            filename=dataset_config["filename"],
            default_subdir=cohort_info.get("default_output_subdir", cohort_info["code"]),
            dataset_type=dataset_type,
            # clinical
            clinical_data_type=dataset_config.get("clinical_data_type", "PATIENT"),
            attribute_ids=dataset_config.get("attribute_ids") or [],
            # molecular / mutations
            molecular_profile_id=dataset_config.get("molecular_profile_id"),
            sample_list_id=dataset_config.get("sample_list_id"),
        )

    # ------------------------------------------------------------------
    #  Build from YAML file
    # ------------------------------------------------------------------

    def build_from_file(self, yaml_file: Path) -> Cohort:
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_file}: {exc}") from exc

        cohort_info = config["cohort"]
        datasets_config = config.get("datasets", [])

        datasets = [
            self._build_dataset(ds_cfg, cohort_info)
            for ds_cfg in datasets_config
        ]

        class ConfiguredCohort(Cohort):
            def __init__(self_inner):
                super().__init__(
                    name=cohort_info["name"],
                    description=cohort_info.get("description", ""),
                    datasets=datasets,
                )

            def download(
                self_inner,
                output_dir: Optional[str] = None,
                download_all: bool = True,
                verbose: bool = True,
                confirm: bool = True,
            ) -> None:
                if output_dir is None:
                    output_dir = (
                        f"data/cbioportal/{cohort_info.get('default_output_subdir', cohort_info['code'])}"
                    )

                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)

                if verbose:
                    print(f"\nDownloading cBioPortal cohort '{cohort_info['code']}' "
                          f"(study: {cohort_info['study_id']}) → {out}")

                datasets_to_download = [
                    ds for ds in self_inner.datasets
                    if not (out / ds.filename).exists()
                ] if download_all else self_inner.datasets

                if not datasets_to_download:
                    if verbose:
                        print("  All files already downloaded.")
                    return

                if confirm:
                    print(f"\n  {len(datasets_to_download)} dataset(s) will be fetched via cBioPortal API:")
                    for ds in datasets_to_download:
                        print(f"    • {ds.filename}  [{ds.description}]")
                    answer = input("\n  Proceed? [Y/n] ").strip().lower()
                    if answer not in ("", "y", "yes"):
                        print("  Cancelled.")
                        return

                for ds in datasets_to_download:
                    print(f"\n  [{ds.name}]")
                    try:
                        ds.download(str(out), confirm=False, verbose=verbose)
                    except Exception as exc:
                        print(f"  ERROR: {exc}")

                if verbose:
                    print(f"\nDone. Files saved to {out}")

        return ConfiguredCohort()

    # ------------------------------------------------------------------
    #  CohortBuilder interface
    # ------------------------------------------------------------------

    def build_cohort(self, cohort_code: str) -> Cohort:
        # Try exact case first, then lower, then upper (mirrors xenabrowser builder)
        for candidate in (
            self.config_dir / f"{cohort_code}.yaml",
            self.config_dir / f"{cohort_code.lower()}.yaml",
            self.config_dir / f"{cohort_code.upper()}.yaml",
        ):
            if candidate.exists():
                return self.build_from_file(candidate)
        raise FileNotFoundError(
            f"No cBioPortal config found for '{cohort_code}' in {self.config_dir}"
        )

    def list_available_cohorts(self) -> List[str]:
        if not self.config_dir.exists():
            return []
        return [f.stem.upper() for f in sorted(self.config_dir.glob("*.yaml"))]
