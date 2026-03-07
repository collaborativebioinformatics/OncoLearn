#!/usr/bin/env python3
"""
Preprocessing utilities for OncoLearn

Includes utilities for splitting manifests, data preprocessing, etc.
"""

import argparse
import random
import sys
from pathlib import Path


def split_tcia_manifest(manifest_path: str, num_splits: int, seed: int = None) -> list[str]:
    """
    Split a TCIA manifest file into multiple non-overlapping manifests.

    Args:
        manifest_path: Path to the original TCIA manifest file
        num_splits: Number of splits to create
        seed: Random seed for reproducibility (optional)

    Returns:
        List of paths to the created manifest files
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if num_splits < 2:
        raise ValueError("Number of splits must be at least 2")

    # Read the manifest file
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    # Parse header and series IDs
    header_lines = []
    series_ids = []
    in_series_list = False

    for line in lines:
        stripped = line.strip()
        if stripped == "ListOfSeriesToDownload=":
            in_series_list = True
            header_lines.append(line)
        elif in_series_list:
            if stripped:  # Non-empty line in series list
                series_ids.append(line)
        else:
            header_lines.append(line)

    print(f"Found {len(series_ids)} series in manifest")

    if len(series_ids) < num_splits:
        raise ValueError(
            f"Cannot split {len(series_ids)} series into {num_splits} parts. "
            f"Number of splits must be <= number of series."
        )

    # Shuffle series IDs for random distribution
    if seed is not None:
        random.seed(seed)
    shuffled_series = series_ids.copy()
    random.shuffle(shuffled_series)

    # Calculate split sizes
    base_size = len(shuffled_series) // num_splits
    remainder = len(shuffled_series) % num_splits

    # Create splits
    split_files = []
    current_idx = 0

    for i in range(num_splits):
        # Determine size of this split
        split_size = base_size + (1 if i < remainder else 0)
        split_series = shuffled_series[current_idx:current_idx + split_size]
        current_idx += split_size

        # Generate output filename
        stem = manifest_path.stem
        suffix = manifest_path.suffix
        parent = manifest_path.parent
        split_filename = f"{stem}_split{i+1:03d}{suffix}"
        split_path = parent / split_filename

        # Write split manifest
        with open(split_path, 'w') as f:
            # Write header
            f.writelines(header_lines)
            # Write series for this split
            f.writelines(split_series)

        split_files.append(str(split_path))
        print(f"Created {split_filename} with {split_size} series")

    return split_files


def register_subcommand(subparsers):
    """Register the preprocess subcommand."""
    parser = subparsers.add_parser(
        "preprocess",
        description="Preprocessing utilities for OncoLearn data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Preprocess data (split manifests, generate K-fold splits, etc.)",
        epilog="""
Examples:
  # Split a TCIA manifest into 4 parts
  oncolearn preprocess --tcia --split 4 --manifest /path/to/manifest.tcia

  # Generate stratified 5-fold training splits from a config
  oncolearn preprocess --kfold --config data/configs/tcga_brca_tabular_only.yaml --n_splits 5

  # Custom output directory and validation fraction
  oncolearn preprocess --kfold --config data/configs/tcga_brca_tabular_only.yaml \\
      --n_splits 5 --val_fraction 0.15 --output_dir data/configs/BRCA/kfold
        """
    )

    # Source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--tcia", action="store_true", help="Preprocess TCIA manifest data")
    source_group.add_argument(
        "--kfold", action="store_true",
        help="Generate stratified K-fold patient-ID split files from a config")

    # --- TCIA-specific arguments ---
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="[--tcia] Path to the manifest file to process")
    parser.add_argument(
        "--split", type=int, default=None,
        help="[--tcia] Number of manifest splits to create (>= 2)")

    # --- K-fold-specific arguments ---
    parser.add_argument(
        "--config", type=str, default=None,
        help="[--kfold] Path to an OncoLearn YAML config")
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="[--kfold] Number of folds (default: 5)")
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="[--kfold] Fraction of train fold reserved for validation (default: 0.1)")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="[--kfold] Root output directory (default: data/configs/<COHORT>/kfold/)")

    # Shared optional arguments
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility")

    parser.set_defaults(func=execute)


def execute(args):
    """Execute the preprocess command."""

    if args.tcia:
        _execute_tcia(args)
    elif args.kfold:
        _execute_kfold(args)


def _execute_tcia(args):
    """Handle the --tcia manifest-split workflow."""
    if not args.manifest:
        print("ERROR: --manifest is required with --tcia")
        sys.exit(1)
    if args.split is None:
        print("ERROR: --split is required with --tcia")
        sys.exit(1)
    if args.split < 2:
        print("ERROR: --split must be at least 2")
        sys.exit(1)

    try:
        print(f"Splitting manifest: {args.manifest}")
        print(f"Number of splits: {args.split}")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")
        print()

        split_files = split_tcia_manifest(args.manifest, args.split, args.seed)

        print()
        print("=" * 80)
        print(f"Successfully created {len(split_files)} manifest files:")
        for f in split_files:
            print(f"  {f}")
        print("=" * 80)

        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _execute_kfold(args):
    """Handle the --kfold stratified split-generation workflow."""
    from pathlib import Path

    from oncolearn.config import load_config
    from oncolearn.registry import get_modality
    from oncolearn.cli.utils.splits import generate_kfold_splits
    import oncolearn.data.modalities  # noqa: F401 — triggers @register_modality decorators

    if not args.config:
        print("ERROR: --config is required with --kfold")
        sys.exit(1)

    try:
        config = load_config(args.config)

        # Find first modality that carries labels (gene or tabular)
        labeled_modality_names = ("gene", "tabular")
        tabular_cfg = next(
            (m for m in config.modalities if m.name in labeled_modality_names), None
        )
        if tabular_cfg is None:
            print("ERROR: No labeled tabular modality ('gene') found in config.")
            sys.exit(1)

        dm_cls = get_modality(tabular_cfg.name)
        dm = dm_cls(**tabular_cfg.kwargs)

        print("Preparing tabular data...")
        dm.prepare_data()
        dm.setup_full()

        full_ds = dm.full_dataset
        patient_ids = full_ds.patient_ids
        if full_ds.labels is None:
            print("ERROR: No labels in tabular dataset — cannot generate stratified splits.")
            sys.exit(1)
        labels = [int(l) for l in full_ds.labels]

        # Derive output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            cohort_code = tabular_cfg.kwargs.get("cohort_code", "dataset")
            cohort_short = cohort_code.replace("TCGA-", "")
            output_dir = Path("data/configs") / cohort_short / "kfold"

        seed = args.seed if args.seed is not None else config.training.seed

        print(
            f"\nGenerating {args.n_splits}-fold splits for {len(patient_ids)} patients"
            f" → {output_dir}\n"
        )
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
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    """Direct entry point for backwards compatibility."""
    parser = argparse.ArgumentParser(
        description="Preprocessing utilities for OncoLearn data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a TCIA manifest into 4 parts
  preprocess --tcia --split 4 --manifest /path/to/manifest.tcia
  
  # Split with a specific random seed for reproducibility
  preprocess --tcia --split 4 --manifest /path/to/manifest.tcia --seed 42
        """
    )

    # Source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--tcia", action="store_true", help="Preprocess TCIA data")

    # Required arguments
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to the manifest file to process")
    parser.add_argument(
        "--split", type=int, required=True,
        help="Number of splits to create (must be >= 2)")

    # Optional arguments
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility")

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
