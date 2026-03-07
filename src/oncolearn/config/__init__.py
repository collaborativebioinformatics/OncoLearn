"""
OncoLearn configuration module.

Canonical usage::

    from oncolearn.config import load_config, OncoLearnConfig

    config = load_config("data/configs/tcga_brca_tabular_only.yaml")
    trainer = OncoTrainer(config=config)
"""

from .schema import (
    DataConfig,
    EncoderConfig,
    ModalityConfig,
    ModelConfig,
    OncoLearnConfig,
    OutputConfig,
    TrainingConfig,
)
from .loader import load_config, save_config

__all__ = [
    "OncoLearnConfig",
    "ModelConfig",
    "DataConfig",
    "EncoderConfig",
    "ModalityConfig",
    "TrainingConfig",
    "OutputConfig",
    "load_config",
    "save_config",
]
