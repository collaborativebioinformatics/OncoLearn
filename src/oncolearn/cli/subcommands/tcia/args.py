"""Shared argument helpers for TCIA subcommands."""

import argparse


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``tcia download``."""
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

    parser.add_argument("--manifest-only", action="store_true",
                        help="Download only manifest files, not images")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to existing manifest file to use for image download")
    parser.add_argument("--unzip", action="store_true", default=False,
                        help="Extract gzipped files after download")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation prompts")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")


def add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``tcia preprocess``."""
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to the TCIA manifest file to split",
    )
    parser.add_argument(
        "--split", type=int, required=True,
        help="Number of manifest splits to create (>= 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
