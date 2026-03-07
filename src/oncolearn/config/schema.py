"""
OncoLearn configuration schema.

Defines the hierarchical dataclass tree that describes a full experiment:
model → data → training → output.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EncoderConfig:
    """Configuration for a single encoder.

    Attributes:
        name: Registry name of the encoder (e.g. ``"gene"``, ``"oncolearn.encoder.multimodal.RNABERTEncoder"``).
              Must match a key registered via :func:`~oncolearn.registry.register_encoder`.
        modality: Dotted modality name used as the batch-routing key (e.g.
                  ``"oncolearn.modality.gene"``).  Must match a ``name`` in
                  ``data.modalities`` when set.  Defaults to ``None``, in which
                  case ``name`` is used as the batch key.
        output_dim: Embedding dimension produced by this encoder.
        kwargs: Encoder-specific keyword arguments forwarded verbatim to the encoder's
                ``__init__`` (e.g. ``checkpoint_path``, ``input_dim``).
    """

    name: str
    modality: Optional[str] = None
    output_dim: int = 128
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityConfig:
    """Configuration for a single data modality.

    Attributes:
        name: Registry name of the modality (e.g. ``"gene"``, ``"oncolearn.modality.gene"``).
        join_on: Patient-ID field used to align multi-modal records.
        join_strategy: Join strategy.  Only ``"inner"`` is currently supported.
        files: List of data file names (relative to the cohort directory) for
               this modality.  Replaces the opaque ``features_files`` / ``clinical_file``
               kwargs.
        kwargs: Remaining modality-specific keyword arguments forwarded to the
                DataModule constructor (e.g. ``n_slices``, per-modality
                ``base_directory`` / ``cohort_code`` overrides).
    """

    name: str
    join_on: str = "patient_id"
    join_strategy: str = "inner"
    files: Optional[List[str]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for all data modalities and their shared settings.

    Attributes:
        modalities: Ordered list of modalities to include.
        base_directory: Root directory for tabular data (e.g. ``"data/xenabrowser"``).
        cohort_code: Cohort identifier (e.g. ``"TCGA-BRCA"``).
        splits_dir: Path to folder with ``train.txt``, ``test.txt``,
                    ``validation.txt`` split files.  When set, overrides
                    per-modality random splits.
    """

    modalities: List[ModalityConfig]
    base_directory: str = "data/xenabrowser"
    cohort_code: str = "TCGA-BRCA"
    splits_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for the fusion model.

    Attributes:
        name: Registry name of the model (e.g. ``"gated_late_fusion"``).
        encoders: Ordered list of encoders to include in the model.
        freeze_encoders: Whether to freeze all pre-trained encoder backbones.
        modality_dropout_prob: Per-modality drop probability during training (0 = disabled).
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
        model: Fusion model settings.  ``model.name`` must match a registered model.
        data: Data settings.  ``data.modalities`` must be non-empty; each
              ``name`` must match a registered modality.

    Optional sections:
        training: Training hyperparameters (all fields have defaults).
        output: Output directory and checkpointing.
    """

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
