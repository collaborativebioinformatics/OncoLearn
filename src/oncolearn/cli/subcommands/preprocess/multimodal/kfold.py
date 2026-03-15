"""K-fold split generation for multimodal cBioPortal data."""

import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set


class _Tee:
    """Mirror writes to two streams simultaneously."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


# --- Label → config path mapping ---

_LABEL_CONFIGS = {
    "stage": "data/configs/modeling/multimodal/tcga_brca_cbioportal_stage.yaml",
    "pam50": "data/configs/modeling/multimodal/tcga_brca_cbioportal_pam50.yaml",
}



def _scan_image_patient_ids(base_dir: str, cohort_code: str) -> Set[str]:
    """Scan a TCIA directory tree and return patient IDs (TCGA-XX-XXXX format)."""
    target = Path(base_dir) / f"TCGA-{cohort_code}"
    if not target.exists():
        print(f"  WARNING: Image directory not found: {target}")
        return set()

    ids: Set[str] = set()
    for file_path in target.rglob("*"):
        for part in file_path.parts:
            if part.startswith("TCGA-"):
                tcga_parts = part.split("-")
                if len(tcga_parts) >= 3:
                    ids.add("-".join(tcga_parts[:3]))
                    break
    return ids


def _generate_train_test_splits(
    patient_ids: List[str],
    labels: List[int],
    output_dir: Path,
    n_splits: int,
    stratified: bool,
    seed: int,
) -> List[Path]:
    """Generate K-fold train/test splits and write them to disk."""
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required: pip install scikit-learn"
        ) from exc

    from oncolearn.cli.utils.splits import write_id_file

    output_dir = Path(output_dir)
    label_map: Dict[str, int] = dict(zip(patient_ids, labels))

    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_ids, labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_ids)

    fold_dirs: List[Path] = []
    for fold_idx, (train_indices, test_indices) in enumerate(split_iter):
        fold_dir = output_dir / f"fold_{fold_idx}"
        train_ids = [patient_ids[i] for i in train_indices]
        test_ids = [patient_ids[i] for i in test_indices]

        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise AssertionError(
                f"fold_{fold_idx} has {len(overlap)} patient(s) in both train and test: "
                f"{sorted(overlap)}"
            )

        write_id_file(fold_dir / "train.txt", train_ids)
        write_id_file(fold_dir / "test.txt", test_ids)

        train_counts = Counter(label_map[pid] for pid in train_ids)
        test_counts = Counter(label_map[pid] for pid in test_ids)
        print(
            f"  fold_{fold_idx}: "
            f"train={len(train_ids)} {dict(sorted(train_counts.items()))}  "
            f"test={len(test_ids)} {dict(sorted(test_counts.items()))}"
        )
        fold_dirs.append(fold_dir)

    return fold_dirs


def kfold(args) -> None:
    """Execute the preprocess multimodal kfold action."""
    from oncolearn.config import load_config
    from oncolearn.data.pipeline.loader import load_pipeline_file, _make_reader
    from oncolearn.data.pipeline.executor import run
    from oncolearn.data.pipeline.nodes import ImageModality, TabularModality

    log_buf = StringIO()
    tee = _Tee(sys.stdout, log_buf)
    sys.stdout = tee  # type: ignore[assignment]

    try:
        # --- Resolve config path from label ---
        config_path = _LABEL_CONFIGS[args.label]
        print(f"Loading config: {config_path}")
        config = load_config(config_path)

        dataset_node = load_pipeline_file(config.data.pipeline)
        modalities = dataset_node.modalities

        # --- Execute each modality to collect patient IDs and labels ---
        modality_ids: Dict[str, Set[str]] = {}
        label_map: Dict[str, int] = {}

        for modality in modalities:
            print(f"  Loading modality '{modality.name}' ...")

            if isinstance(modality, ImageModality):
                ids = _scan_image_patient_ids(modality.base_dir, modality.cohort_code)
                if ids:
                    modality_ids[modality.name] = ids
                    print(f"    {len(ids)} patients (scanned from disk)")
                else:
                    print(f"  WARNING: No image patients found for '{modality.name}'")
                continue

            if isinstance(modality, TabularModality):
                try:
                    reader = _make_reader(modality)
                    df = run(modality.pipeline, reader)
                except Exception as e:
                    print(f"  WARNING: Could not load modality '{modality.name}': {e}")
                    continue

                if "patient_id" not in df.columns:
                    print(
                        f"  WARNING: Modality '{modality.name}' has no 'patient_id' column."
                    )
                    continue

                ids = set(df["patient_id"].dropna().tolist())
                modality_ids[modality.name] = ids
                print(f"    {len(ids)} patients")

                if modality.label_col and modality.label_col in df.columns:
                    transform = modality.label_transform or (lambda x: x)
                    for _, row in df.iterrows():
                        raw = row[modality.label_col]
                        label = transform(raw)
                        if label is not None:
                            try:
                                label_map[row["patient_id"]] = int(label)
                            except (ValueError, TypeError):
                                pass
                    print(
                        f"    {len(label_map)} patients with labels "
                        f"from '{modality.label_col}'"
                    )

        if not modality_ids:
            print("ERROR: No modality data could be loaded.")
            sys.exit(1)

        # --- Intersect patient IDs across all modalities ---
        candidate_ids: Set[str] = None  # type: ignore[assignment]
        for name, ids in modality_ids.items():
            if candidate_ids is None:
                candidate_ids = ids
            else:
                candidate_ids = candidate_ids & ids

        if not candidate_ids:
            print("ERROR: No patients in the multimodal intersection.")
            sys.exit(1)

        print(f"  Multimodal intersection: {len(candidate_ids)} patients")

        if not label_map:
            print("ERROR: No labels found — cannot generate splits.")
            sys.exit(1)

        patient_ids = sorted(pid for pid in candidate_ids if pid in label_map)
        labels = [label_map[pid] for pid in patient_ids]
        print(f"  Patients with valid labels: {len(patient_ids)}")

        if not patient_ids:
            print("ERROR: No patients with valid labels in the intersection.")
            sys.exit(1)

        # --- Resolve output directory ---
        output_dir = (
            Path(args.output)
            if args.output
            else Path(
                f"data/configs/modeling/multimodal/splits/{args.label}/kfold"
            )
        )

        kind = "StratifiedKFold" if args.stratified else "KFold"
        print(
            f"\nGenerating {args.n_splits}-fold ({kind}) splits "
            f"for {len(patient_ids)} patients → {output_dir}\n"
        )

        fold_dirs = _generate_train_test_splits(
            patient_ids=patient_ids,
            labels=labels,
            output_dir=output_dir,
            n_splits=args.n_splits,
            stratified=args.stratified,
            seed=args.seed,
        )

        print(f"\nSaved {len(fold_dirs)} folds to {output_dir}")
        print("To use fold 0, add to your config YAML:")
        print(f"  splits_dir: {fold_dirs[0]}")

        sys.stdout = sys.__stdout__
        log_path = output_dir / "log.txt"
        log_path.write_text(log_buf.getvalue())
        print(f"Log written to {log_path}")

        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        sys.stdout = sys.__stdout__
