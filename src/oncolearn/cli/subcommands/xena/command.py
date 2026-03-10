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

    from .args import add_download_arguments, add_preprocess_arguments
    from .download import download
    from .preprocess import preprocess as preprocess_execute

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

    pp_parser = xena_sub.add_parser(
        "preprocess",
        description="Generate stratified K-fold patient-ID split files from a config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate K-fold training splits from Xena tabular data",
        epilog="""
Examples:
  oncolearn xena preprocess --config data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml
  oncolearn xena preprocess --config data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml --n_splits 10 --seed 42
        """,
    )
    add_preprocess_arguments(pp_parser)
    pp_parser.set_defaults(func=preprocess_execute)


    parser.set_defaults(func=lambda args: parser.print_help())
