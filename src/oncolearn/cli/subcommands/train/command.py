"""Train subcommand — register and execute."""

import argparse
import sys

from .args import add_arguments, _VARIANT_CONFIGS


def register_subcommand(subparsers) -> None:
    """Register the ``train`` subcommand with *subparsers*."""
    parser = subparsers.add_parser(
        "train",
        description="Train an OncoLearn model from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Train a model",
        epilog="""
Examples:
  # Config-driven (recommended)
  oncolearn train --config data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml

  # Quick shorthand
  oncolearn train --variant v2_no_imaging --epochs 10 --batch_size 8
        """,
    )
    add_arguments(parser)
    parser.set_defaults(func=execute)


def execute(args) -> None:
    """Execute the train command from parsed *args*."""
    from oncolearn.config import load_config
    from oncolearn.trainer import OncoTrainer

    config_path = args.config or _VARIANT_CONFIGS[args.variant]
    config = load_config(config_path)

    # CLI overrides only apply when using the shorthand variant (no explicit config)
    if not args.config:
        config.training.max_epochs = args.epochs
        config.training.batch_size = args.batch_size

    trainer = OncoTrainer(config)
    trainer.train()
    sys.exit(0)


def main(argv=None) -> None:
    """Standalone entry point (used by ``python -m oncolearn.trainer``)."""
    parser = argparse.ArgumentParser(
        description="OncoLearn trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m oncolearn.trainer --config data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml
  python -m oncolearn.trainer --variant v2_no_imaging --epochs 10 --batch_size 8
        """,
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    execute(args)
