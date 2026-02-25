#!/usr/bin/env python3
"""
TCIA data preprocessing utilities.

This module contains functions for preprocessing TCIA (The Cancer Imaging Archive) data,
including manifest splitting and other preprocessing operations.
"""

import random
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

    Raises:
        FileNotFoundError: If the manifest file doesn't exist
        ValueError: If num_splits is less than 2 or greater than the number of series
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
