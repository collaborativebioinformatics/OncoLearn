"""Multimodal preprocessing subcommand group."""

import argparse


def register_subcommand(subparsers) -> None:
    """Register the ``multimodal`` subcommand group with *subparsers*."""
    parser = subparsers.add_parser(
        "multimodal",
        description="Preprocessing commands for multimodal (cBioPortal + TCIA) data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Multimodal preprocessing commands",
    )

    multi_sub = parser.add_subparsers(
        title="multimodal commands",
        dest="multimodal_command",
        help="Multimodal sub-command to run",
    )

    from .args import add_kfold_arguments
    from .kfold import kfold

    kfold_parser = multi_sub.add_parser(
        "kfold",
        description="Generate K-fold train/test patient-ID split files from multimodal cBioPortal data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate K-fold splits for multimodal data",
        epilog="""
Examples:
  oncolearn preprocess multimodal kfold 5 --stratified --label stage
  oncolearn preprocess multimodal kfold 5 --label pam50 --seed 42
  oncolearn preprocess multimodal kfold 5 --label stage --output /tmp/splits
        """,
    )
    add_kfold_arguments(kfold_parser)
    kfold_parser.set_defaults(func=kfold)

    parser.set_defaults(func=lambda args: parser.print_help())
