"""Argument definitions for the preprocess multimodal kfold subcommand."""

import argparse


def add_kfold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for ``preprocess multimodal kfold``."""
    parser.add_argument(
        "n_splits",
        type=int,
        nargs="?",
        default=5,
        help="Number of KFold splits (default: 5)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=False,
        help="Use StratifiedKFold (default: KFold without stratification)",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=["pam50", "stage"],
        help="Label task: 'pam50' or 'stage'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for split files "
            "(default: data/configs/modeling/multimodal/splits/<label>/kfold)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
