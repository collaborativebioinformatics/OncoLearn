"""Argument definitions for the xena download subcommand."""

import argparse


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``xena download``."""
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--cohorts", type=str,
        help="Cohort code(s), comma-separated (e.g., BRCA,LUAD)",
    )
    action_group.add_argument(
        "--all", action="store_true",
        help="Download all available cohorts",
    )
    action_group.add_argument(
        "--list", action="store_true",
        help="List available cohorts and exit",
    )

    parser.add_argument("--category", type=str, default=None,
                        help="Filter datasets by category (e.g., mirna, mrna, mutation)")
    parser.add_argument("--ids", type=str, default=None,
                        help="Specific dataset ID(s) to download, comma-separated")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--unzip", action="store_true", default=False,
                        help="Extract gzipped files after download")
    parser.add_argument("--mapping", action="store_true", default=False,
                        help="Download gene mapping files")
    parser.add_argument("--raw", action="store_true", default=False,
                        help="Download raw data files")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation prompts")


def add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``xena preprocess``."""
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to an OncoLearn YAML config",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of train fold reserved for validation (default: 0.1)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (default: data/configs/modeling/multimodal/splits/kfold/)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
