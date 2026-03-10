"""Xena K-fold split generation logic."""

import re
import sys
from pathlib import Path


def _load_xenabrowser_tsv(file_path: Path):
    """Load a XenaBrowser TSV, auto-transposing genomic-matrix files."""
    import pandas as pd

    df = pd.read_csv(str(file_path), sep="\t", low_memory=False)
    sample_cols = [c for c in df.columns[1:6] if isinstance(c, str) and c.startswith("TCGA-")]
    if len(sample_cols) >= 3:
        id_col = df.columns[0]
        df = df.set_index(id_col).T.reset_index()
        df = df.rename(columns={"index": "patient_id"})
    elif "sample" in df.columns:
        df = df.rename(columns={"sample": "patient_id"})
    elif df.columns[0] == "Unnamed: 0":
        df = df.rename(columns={"Unnamed: 0": "patient_id"})
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].apply(
            lambda x: x[:12] if isinstance(x, str) and x.startswith("TCGA") else x
        )
    return df


_STAGE_PATTERNS = [
    (re.compile(r"stage\s*i(?!v|i)", re.I), 0),
    (re.compile(r"stage\s*ii(?!i)", re.I), 1),
    (re.compile(r"stage\s*iii", re.I), 2),
    (re.compile(r"stage\s*iv", re.I), 3),
]


def _map_stage(raw) -> "int | None":
    if not isinstance(raw, str):
        return None
    for pat, lbl in _STAGE_PATTERNS:
        if pat.search(raw):
            return lbl
    return None


def preprocess(args) -> None:
    """Execute the xena preprocess (k-fold) action."""
    from oncolearn.config import load_config
    from oncolearn.cli.utils.splits import generate_kfold_splits

    try:
        config = load_config(args.config)
        data_cfg = config.data

        # --- Gene/tabular modality: collect all patient IDs ---
        tabular_cfg = next(
            (m for m in data_cfg.modalities if "gene" in m.name or "tabular" in m.name),
            None,
        )
        if tabular_cfg is None:
            print("ERROR: No labeled tabular modality ('gene') found in config.")
            sys.exit(1)

        gene_files = tabular_cfg.files or ["TCGA-BRCA.mirna.tsv", "pam50.tsv"]
        gene_dir = Path(
            tabular_cfg.kwargs.get("base_directory", data_cfg.base_directory)
        ) / tabular_cfg.kwargs.get("cohort_code", data_cfg.cohort_code)

        gene_ids: set = set()
        for fname in gene_files:
            fpath = gene_dir / fname
            if fpath.exists() and fpath.suffix.lower() == ".tsv":
                df = _load_xenabrowser_tsv(fpath)
                if "patient_id" in df.columns:
                    gene_ids.update(df["patient_id"].dropna().tolist())

        print(f"  Gene/tabular patients: {len(gene_ids)}")

        # --- Image modality: collect patient IDs from image directory ---
        image_cfg = next(
            (m for m in data_cfg.modalities if "image" in m.name), None
        )
        image_ids: set = set()
        if image_cfg is not None:
            img_base = image_cfg.kwargs.get("base_directory", "data/tcia")
            img_cohort = image_cfg.kwargs.get("cohort_code", data_cfg.cohort_code.replace("TCGA-", ""))
            img_dir = Path(img_base) / f"TCGA-{img_cohort}"
            if img_dir.exists():
                for entry in img_dir.rglob("*"):
                    if entry.is_dir() and entry.name.startswith("TCGA-"):
                        parts = entry.name.split("-")
                        if len(parts) >= 3:
                            image_ids.add("-".join(parts[:3]))
            print(f"  Image patients: {len(image_ids)}")

        # --- Clinical modality: get stage labels for stratification ---
        clinical_cfg = next(
            (m for m in data_cfg.modalities if "clinical" in m.name), None
        )
        clinical_label_map: dict = {}
        if clinical_cfg is not None:
            clin_file = (clinical_cfg.files or ["TCGA-BRCA.clinical.tsv"])[0]
            clin_base = clinical_cfg.kwargs.get("base_directory", data_cfg.base_directory)
            clin_cohort = clinical_cfg.kwargs.get("cohort_code", data_cfg.cohort_code)
            clin_path = Path(clin_base) / clin_cohort / clin_file
            stage_col = clinical_cfg.kwargs.get("stage_col", "ajcc_pathologic_stage.diagnoses")
            if clin_path.exists():
                clin_df = _load_xenabrowser_tsv(clin_path)
                if stage_col in clin_df.columns and "patient_id" in clin_df.columns:
                    clin_df["label"] = clin_df[stage_col].apply(_map_stage)
                    for _, row in clin_df.dropna(subset=["label"]).iterrows():
                        try:
                            clinical_label_map[row["patient_id"]] = int(row["label"])
                        except (ValueError, TypeError):
                            pass
            print(f"  Clinical patients with stage labels: {len(clinical_label_map)}")

        # --- Build multimodal intersection ---
        candidate_ids = gene_ids
        if image_ids:
            candidate_ids = candidate_ids & image_ids
            print(f"  Gene ∩ Image intersection: {len(candidate_ids)}")

        if clinical_label_map:
            patient_ids = sorted(pid for pid in candidate_ids if pid in clinical_label_map)
            labels = [clinical_label_map[pid] for pid in patient_ids]
            print(f"  Patients with clinical stage labels: {len(patient_ids)}")
        else:
            print("ERROR: No clinical labels found — cannot generate stratified splits.")
            sys.exit(1)

        if not patient_ids:
            print("ERROR: No patients found in multimodal intersection with valid labels.")
            sys.exit(1)

        # --- Derive output directory ---
        output_dir = Path(args.output_dir) if args.output_dir else Path("data/configs/modeling/multimodal/splits/kfold")

        seed = args.seed if args.seed is not None else config.training.seed

        print(f"\nGenerating {args.n_splits}-fold splits for {len(patient_ids)} patients → {output_dir}\n")
        fold_dirs = generate_kfold_splits(
            patient_ids=patient_ids,
            labels=labels,
            output_dir=output_dir,
            n_splits=args.n_splits,
            val_fraction=args.val_fraction,
            seed=seed,
        )

        print(f"\nSaved {len(fold_dirs)} folds to {output_dir}")
        print("To use fold 0, add to your config YAML:")
        print(f"  splits_dir: {fold_dirs[0]}")

        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(1)
