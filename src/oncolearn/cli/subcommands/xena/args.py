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
