#!/usr/bin/env python3
"""
CLI interface for OncoLearn data preprocessing.

This module provides command-line interface for preprocessing operations.
"""

import argparse
import sys

from oncolearn.api.tcia.preprocessing import split_tcia_manifest


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
