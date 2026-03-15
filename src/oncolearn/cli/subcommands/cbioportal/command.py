"""cBioPortal data-source subcommand group."""

import argparse


def register_subcommand(subparsers) -> None:
    """Register the ``cbioportal`` subcommand group with *subparsers*."""
    parser = subparsers.add_parser(
        "cbioportal",
        description="Download cancer data from cBioPortal via the REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="cBioPortal data commands",
        epilog="""
Examples:
  oncolearn cbioportal download --cohorts BRCA
  oncolearn cbioportal download --list
  oncolearn cbioportal download --list --search breast --cancer-type brca
  oncolearn cbioportal download --all --yes
        """,
    )

    cbio_sub = parser.add_subparsers(
        title="cbioportal commands",
        dest="cbioportal_command",
        help="cBioPortal sub-command to run",
    )

    from .args import add_download_arguments
    from .download import download

    dl_parser = cbio_sub.add_parser(
        "download",
        description="Download cohorts from cBioPortal via the REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Download cBioPortal data",
        epilog="""
Examples:
  oncolearn cbioportal download --cohorts BRCA
  oncolearn cbioportal download --list --search breast
  oncolearn cbioportal download --all --yes
        """,
    )
    add_download_arguments(dl_parser)
    dl_parser.set_defaults(func=download)

    parser.set_defaults(func=lambda args: parser.print_help())
