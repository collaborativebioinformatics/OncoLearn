"""Xena Browser data-source subcommand group."""

import argparse


def register_subcommand(subparsers) -> None:
    """Register the ``xena`` subcommand group with *subparsers*."""
    parser = subparsers.add_parser(
        "xena",
        description="Download cancer data from UCSC Xena Browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="UCSC Xena Browser data commands",
        epilog="""
Examples:
  oncolearn xena download --cohorts BRCA
  oncolearn xena download --cohorts BRCA --unzip
  oncolearn xena download --list
  oncolearn xena download --cohorts BRCA --category mirna
        """,
    )

    xena_sub = parser.add_subparsers(
        title="xena commands",
        dest="xena_command",
        help="Xena sub-command to run",
    )

    from .args import add_download_arguments
    from .download import download

    dl_parser = xena_sub.add_parser(
        "download",
        description="Download cohorts from UCSC Xena Browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Download Xena Browser data",
        epilog="""
Examples:
  oncolearn xena download --cohorts BRCA
  oncolearn xena download --cohorts BRCA --unzip
  oncolearn xena download --list --category mirna
  oncolearn xena download --all --yes
        """,
    )
    add_download_arguments(dl_parser)
    dl_parser.set_defaults(func=download)

    parser.set_defaults(func=lambda args: parser.print_help())
