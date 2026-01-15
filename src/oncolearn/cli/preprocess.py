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
        help="Preprocess data (split manifests, etc.)",
        epilog="""
Examples:
  # Split a TCIA manifest into 4 parts
  oncolearn preprocess --tcia --split 4 --manifest /path/to/manifest.tcia
  
  # Split with a specific random seed for reproducibility
  oncolearn preprocess --tcia --split 4 --manifest /path/to/manifest.tcia --seed 42
        """
    )

    # Source selection (for future extensibility)
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

    # Set the function to call when this subcommand is used
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the preprocess command."""

    if not args.tcia:
        print("ERROR: Currently only --tcia preprocessing is supported")
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
