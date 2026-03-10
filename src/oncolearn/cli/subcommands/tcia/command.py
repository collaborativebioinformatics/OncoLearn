"""TCIA data-source subcommand group — registers download + preprocess sub-subcommands."""

import argparse


def register_subcommand(subparsers) -> None:
    """Register the ``tcia`` subcommand group with *subparsers*."""
    parser = subparsers.add_parser(
        "tcia",
        description="Download and preprocess TCIA imaging data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="TCIA imaging data commands",
        epilog="""
Examples:
  oncolearn tcia download --cohorts BRCA
  oncolearn tcia download --list
  oncolearn tcia preprocess --manifest manifest.tcia --split 4
        """,
    )

    tcia_sub = parser.add_subparsers(
        title="tcia commands",
        dest="tcia_command",
        help="TCIA sub-command to run",
    )

    from .args import add_download_arguments, add_preprocess_arguments
    from .download import download
    from .preprocess import preprocess

    # Register download sub-subcommand
    dl_parser = tcia_sub.add_parser(
        "download",
        description="Download TCIA imaging manifests and/or images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Download TCIA imaging data",
        epilog="""
Examples:
  oncolearn tcia download --cohorts BRCA
  oncolearn tcia download --all --yes
  oncolearn tcia download --list
        """,
    )
    add_download_arguments(dl_parser)
    dl_parser.set_defaults(func=download)

    # Register preprocess sub-subcommand
    pp_parser = tcia_sub.add_parser(
        "preprocess",
        description="Split a TCIA manifest file into multiple non-overlapping manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Split TCIA manifest files",
        epilog="""
Examples:
  oncolearn tcia preprocess --manifest manifest.tcia --split 4
  oncolearn tcia preprocess --manifest manifest.tcia --split 4 --seed 42
        """,
    )
    add_preprocess_arguments(pp_parser)
    pp_parser.set_defaults(func=preprocess)

    parser.set_defaults(func=lambda args: parser.print_help())
