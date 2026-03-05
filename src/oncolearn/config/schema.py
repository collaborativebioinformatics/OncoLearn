"""
OncoLearn configuration schema.

Defines the hierarchical dataclass tree that describes a full experiment:
model → modalities → training → output.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EncoderConfig:
    """Configuration for a single encoder.

    Attributes:
        name: Registry name of the encoder (e.g. ``"gene"``, ``"image"``, ``"tabular"``).
              Must match a key registered via :func:`~oncolearn.registry.register_encoder`.
        output_dim: Embedding dimension produced by this encoder.
        kwargs: Encoder-specific keyword arguments forwarded verbatim to the encoder's
                ``__init__`` (e.g. ``checkpoint_path``, ``input_dim``).
    """

    name: str
    output_dim: int = 128
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityConfig:
    """Configuration for a single data modality.

    Attributes:
        name: Registry name of the modality (e.g. ``"tabular"``, ``"image"``).
              Must match a key registered via :func:`~oncolearn.registry.register_modality`.
        kwargs: Keyword arguments forwarded verbatim to the modality's
                ``DataModule.__init__``.  Use this to set cohort codes, file lists,
                slice counts, etc.
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for the fusion model.

    Attributes:
        name: Registry name of the model (e.g. ``"gated_late_fusion"``).
              Must match a key registered via :func:`~oncolearn.registry.register_model`.
        encoders: Ordered list of encoders to include in the model. Each entry specifies
                  the encoder name (registry key), its output dimension, and any
                  encoder-specific kwargs (e.g. ``checkpoint_path`` for the image encoder).
        freeze_encoders: Whether to freeze all pre-trained encoder backbones.
        modality_dropout_prob: Per-modality drop probability during training (0 = disabled).
                               At least one modality is always retained.
    """

    name: str
    encoders: List[EncoderConfig] = field(default_factory=list)
    num_stage_classes: int = 5
    num_subtype_classes: int = 0
    freeze_encoders: bool = True
    dropout: float = 0.2
    modality_dropout_prob: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    num_workers: int = 4
    accelerator: str = "auto"
    devices: int = 1
    early_stopping_patience: int = 10
    subtype_lambda: float = 0.3
    scheduler: str = "cosine"
    seed: int = 42
    use_class_weights: bool = True


@dataclass
class OutputConfig:
    """Output directory and checkpointing settings."""

    dir: str = "outputs"
    experiment_name: str = "experiment"
    save_every_n_epochs: int = 5


@dataclass
class OncoLearnConfig:
    """Top-level experiment configuration.

    Required sections:
        model: Fusion model settings. ``model.name`` must match a registered model.
        modalities: At least one modality. Each ``name`` must match a registered modality.

    Optional sections:
        training: Training hyperparameters (all fields have defaults).
        output: Output directory and checkpointing.
        join_on: Patient-ID field used to align multi-modal records (default: ``"patient_id"``).
        join_strategy: How to join modalities. Only ``"inner"`` is currently supported.
    """

    model: ModelConfig
    modalities: List[ModalityConfig]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    join_on: str = "patient_id"
    join_strategy: str = "inner"
    splits_dir: Optional[str] = None
