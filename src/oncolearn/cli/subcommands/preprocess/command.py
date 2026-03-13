"""Preprocess subcommand group."""

import argparse


def register_subcommand(subparsers) -> None:
    """Register the ``preprocess`` subcommand group with *subparsers*."""
    parser = subparsers.add_parser(
        "preprocess",
        description="Data preprocessing commands (split generation, etc.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Data preprocessing commands",
        epilog="""
Examples:
  oncolearn preprocess multimodal kfold 5 --stratified --label stage
  oncolearn preprocess multimodal kfold 5 --label pam50 --seed 42
        """,
    )

    pre_sub = parser.add_subparsers(
        title="preprocess commands",
        dest="preprocess_command",
        help="Preprocess sub-command to run",
    )

    from .multimodal.command import register_subcommand as register_multimodal

    register_multimodal(pre_sub)

    parser.set_defaults(func=lambda args: parser.print_help())
