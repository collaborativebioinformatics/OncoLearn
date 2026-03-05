#!/usr/bin/env python3
"""
OncoLearn training CLI.

Wraps :class:`~oncolearn.trainer.OncoTrainer` for use via the ``oncolearn train``
command or directly as ``python -m oncolearn.trainer``.
"""

import argparse
import sys

_VARIANT_CONFIGS = {
    "v1_imaging":    "data/configs/tcga_brca_multimodal.yaml",
    "v2_no_imaging": "data/configs/tcga_brca_tabular_only.yaml",
}


def register_subcommand(subparsers):
    """Register the ``train`` subcommand with *subparsers*."""
    parser = subparsers.add_parser(
        "train",
        description="Train an OncoLearn model from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Train a model",
        epilog="""
Examples:
  # Config-driven (recommended)
  oncolearn train --config data/configs/tcga_brca_tabular_only.yaml

  # Quick shorthand
  oncolearn train --variant v2_no_imaging --epochs 10 --batch_size 8
        """,
    )
    _add_arguments(parser)
    parser.set_defaults(func=execute)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to an OncoLearn YAML config (data/configs/*.yaml).",
    )
    parser.add_argument(
        "--variant", type=str, default="v2_no_imaging",
        choices=list(_VARIANT_CONFIGS),
        help="Quick shorthand used when --config is not provided.",
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max training epochs (shorthand override).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (shorthand override).")


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


def main(argv=None) -> None:
    """Standalone entry point (used by ``python -m oncolearn.trainer``)."""
    parser = argparse.ArgumentParser(
        description="OncoLearn trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m oncolearn.trainer --config data/configs/tcga_brca_tabular_only.yaml
  python -m oncolearn.trainer --variant v2_no_imaging --epochs 10 --batch_size 8
        """,
    )
    _add_arguments(parser)
    args = parser.parse_args(argv)
    execute(args)


if __name__ == "__main__":
    main()
