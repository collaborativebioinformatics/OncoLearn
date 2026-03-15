"""Argument definitions for the train subcommand."""

import argparse

_VARIANT_CONFIGS = {
    "v1_imaging":    "data/configs/modeling/multimodal/tcga_brca_multimodal.yaml",
    "v2_no_imaging": "data/configs/modeling/multimodal/tcga_brca_tabular_only.yaml",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to an OncoLearn YAML config (data/configs/modeling/multimodal/*.yaml).",
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
