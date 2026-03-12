"""Xena K-fold split generation logic (pipeline-based)."""

import sys
from pathlib import Path


def preprocess(args) -> None:
    """Execute the xena preprocess (k-fold) action using the pipeline DSL."""
    from oncolearn.config import load_config
    from oncolearn.cli.utils.splits import generate_kfold_splits
    from oncolearn.data.pipeline.loader import load_pipeline_file, _make_reader
    from oncolearn.data.pipeline.executor import run

    try:
        config = load_config(args.config)
        data_cfg = config.data

        dataset_node = load_pipeline_file(data_cfg.pipeline)
        modalities = dataset_node.modalities

        # --- Execute each modality pipeline to collect patient IDs ---
        modality_ids: dict = {}
        clinical_label_map: dict = {}

        for modality in modalities:
            print(f"  Loading modality '{modality.name}' ...")
            try:
                reader = _make_reader(modality)
                df = run(modality.pipeline, reader)
            except Exception as e:
                print(f"  WARNING: Could not load modality '{modality.name}': {e}")
                continue

            if "patient_id" not in df.columns:
                print(f"  WARNING: Modality '{modality.name}' has no 'patient_id' column.")
                continue

            ids = set(df["patient_id"].dropna().tolist())
            modality_ids[modality.name] = ids
            print(f"    {len(ids)} patients")

            # If this modality has a label_col, extract stage labels for stratification
            if modality.label_col and modality.label_col in df.columns:
                transform = modality.label_transform or (lambda x: x)
                for _, row in df.iterrows():
                    raw = row[modality.label_col]
                    label = transform(raw)
                    if label is not None:
                        try:
                            clinical_label_map[row["patient_id"]] = int(label)
                        except (ValueError, TypeError):
                            pass
                print(f"    {len(clinical_label_map)} patients with labels from '{modality.label_col}'")

        if not modality_ids:
            print("ERROR: No modality data could be loaded.")
            sys.exit(1)

        # --- Build intersection of all modality patient sets ---
        candidate_ids = None
        for name, ids in modality_ids.items():
            if candidate_ids is None:
                candidate_ids = ids
            else:
                candidate_ids = candidate_ids & ids

        if not candidate_ids:
            print("ERROR: No patients in the multimodal intersection.")
            sys.exit(1)

        print(f"  Multimodal intersection: {len(candidate_ids)} patients")

        if not clinical_label_map:
            print("ERROR: No clinical labels found — cannot generate stratified splits.")
            sys.exit(1)

        patient_ids = sorted(pid for pid in candidate_ids if pid in clinical_label_map)
        labels = [clinical_label_map[pid] for pid in patient_ids]
        print(f"  Patients with stage labels: {len(patient_ids)}")

        if not patient_ids:
            print("ERROR: No patients with valid labels in the intersection.")
            sys.exit(1)

        # --- Derive output directory ---
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path("data/configs/modeling/multimodal/splits/kfold")
        )

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
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(1)
