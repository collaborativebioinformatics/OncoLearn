"""Argument definitions for the cbioportal download subcommand."""

import argparse


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``cbioportal download``."""
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--cohorts", type=str,
        help="Cohort code(s), comma-separated (e.g., BRCA,LUAD)",
    )
    action_group.add_argument(
        "--all", action="store_true",
        help="Download all configured cohorts",
    )
    action_group.add_argument(
        "--list", action="store_true",
        help="List available cohorts and exit",
    )

    parser.add_argument("--search", type=str, default=None,
                        help="Search keyword for cBioPortal study discovery (used with --list)")
    parser.add_argument("--cancer-type", type=str, default=None,
                        help="Filter cBioPortal studies by cancer type ID (e.g. 'brca') (used with --list)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation prompts")
