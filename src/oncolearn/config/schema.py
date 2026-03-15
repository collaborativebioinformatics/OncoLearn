"""
OncoLearn configuration schema.

Defines the hierarchical dataclass tree that describes a full experiment:
model → data → training → output.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class EncoderConfig:
    """Configuration for a single encoder.

    Attributes:
        name: Registry name of the encoder (e.g. ``"oncolearn.encoder.multimodal.RNABERTEncoder"``).
              Must match a key registered via :func:`~oncolearn.registry.register_encoder`.
        modality: Dotted modality name used as the batch-routing key (e.g.
                  ``"oncolearn.modality.gene"``).  Must match a modality ``name``
                  defined in the pipeline file when set.  Defaults to ``None``,
                  in which case ``name`` is used as the batch key.
        output_dim: Embedding dimension produced by this encoder.
        kwargs: Encoder-specific keyword arguments forwarded verbatim to the encoder's
                ``__init__`` (e.g. ``checkpoint_path``, ``input_dim``).
    """

    name: str
    modality: Optional[str] = None
    output_dim: int = 128
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for the data pipeline.

    Attributes:
        pipeline: Path to a pipeline ``.py`` file that defines a
                  :class:`~oncolearn.data.pipeline.Dataset` node.
                  E.g. ``"data/configs/modeling/multimodal/preprocessing/tcga_brca_cbioportal.py"``.
        splits_dir: Path to a folder with ``train.txt``, ``test.txt``,
                    ``validation.txt`` split files.  When set, overrides
                    per-modality random splits inside
                    :class:`~oncolearn.data.multimodal.MultimodalDataModule`.
    """

    pipeline: str
    splits_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for the fusion model.

    Attributes:
        name: Registry name of the model (e.g. ``"oncolearn.model.multimodal.gated_late_fusion"``).
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
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: str = "torch.optim.AdamW"
    params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-4, "weight_decay": 1e-5})


@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""

    name: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    params: Dict[str, Any] = field(default_factory=dict)
    monitor: str = "val_loss"
    interval: str = "epoch"
    frequency: int = 1


@dataclass
class LossConfig:
    """Configuration for the loss function."""

    name: str = "torch.nn.CrossEntropyLoss"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegularizationConfig:
    """Regularization settings."""

    l1_lambda: float = 0.0
    gradient_clip_val: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class HpoParamSpec:
    """Search range for a single hyperparameter.

    The *type* field controls which Optuna ``suggest_*`` method is used:

    * ``"float"`` → :meth:`optuna.Trial.suggest_float` (use ``low``/``high``; ``log`` optional)
    * ``"int"``   → :meth:`optuna.Trial.suggest_int`   (use ``low``/``high``; ``step`` optional)
    * ``"categorical"`` → :meth:`optuna.Trial.suggest_categorical` (use ``choices``)

    The parameter is applied to the config via its dotted path (the dict key in
    ``HpoConfig.search_space``), e.g. ``"training.optimizer.params.lr"``.
    Dict segments (like ``params``) and list indices (like ``model.encoders.0``)
    are handled automatically.
    """

    type: str  # "float" | "int" | "categorical"
    # float / int
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[Union[int, float]] = None
    # categorical
    choices: Optional[List[Any]] = None


@dataclass
class HpoConfig:
    """Optional hyperparameter optimisation settings.

    When ``training.hpo`` is present in the YAML config, :class:`~oncolearn.trainer.OncoTrainer`
    will run an Optuna study before the final training run.  The best found parameters
    are applied to the config and returned alongside the normal training metrics.

    ``search_space`` accepts flat dotted-path keys as well as two special nested keys:

    * ``training.optimizers`` — maps optimizer class name → per-param specs.  When
      multiple optimizers are listed, Optuna also searches over which one to use.
    * ``training.losses`` — same pattern for the loss function.
    * ``training.schedulers`` — same pattern for the LR scheduler.

    Example YAML::

        training:
          hpo:
            n_trials: 30
            epochs_per_trial: 10
            metric: val_acc
            search_space:
              training.optimizers:
                torch.optim.AdamW:
                  lr:           {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
                  weight_decay: {type: float, low: 1.0e-6, high: 1.0e-3, log: true}
                torch.optim.SGD:
                  lr:       {type: float, low: 1.0e-3, high: 1.0e-1, log: true}
                  momentum: {type: float, low: 0.5, high: 0.99}
              training.losses:
                torch.nn.CrossEntropyLoss:
                  label_smoothing: {type: float, low: 0.0, high: 0.3}
              training.batch_size:
                type: categorical
                choices: [4, 8, 16, 32]
              model.dropout:
                type: float
                low: 0.05
                high: 0.5
    """

    n_trials: int = 20
    study_name: str = "oncolearn_hpo"
    storage: Optional[str] = None
    direction: str = "maximize"
    metric: str = "val_acc"
    pruning: bool = True
    epochs_per_trial: Optional[int] = None
    seed: int = 42
    search_space: Dict[str, HpoParamSpec] = field(default_factory=dict)
    optimizers: Dict[str, Dict[str, HpoParamSpec]] = field(default_factory=dict)
    """Per-optimizer conditional param search spaces.

    Maps optimizer dotted class name → {param_name → HpoParamSpec}.  When
    multiple optimizers are listed, a categorical choice is also suggested.
    Only the params for the *chosen* optimizer are sampled in each trial,
    preventing invalid kwargs (e.g. ``momentum`` passed to AdamW).
    """
    losses: Dict[str, Dict[str, HpoParamSpec]] = field(default_factory=dict)
    """Per-loss conditional param search spaces.

    Maps loss dotted class name → {param_name → HpoParamSpec}.  When
    multiple losses are listed, a categorical choice is also suggested.
    """
    schedulers: Dict[str, Dict[str, HpoParamSpec]] = field(default_factory=dict)
    """Per-scheduler conditional param search spaces.

    Maps scheduler dotted class name → {param_name → HpoParamSpec}.  When
    multiple schedulers are listed, a categorical choice is also suggested.
    Only the params for the *chosen* scheduler are sampled in each trial.
    ``training.scheduler.name`` and ``training.scheduler.params`` are updated;
    ``monitor``, ``interval``, and ``frequency`` are left unchanged.
    """


@dataclass
class CrossValidationConfig:
    """Cross-validation settings for HPO.

    When enabled, each HPO trial trains on every fold in ``folds_dirs``
    and the trial metric is the mean across folds.
    """

    enabled: bool = False
    folds_dirs: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    max_epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    accelerator: str = "auto"
    devices: int = 1
    early_stopping_patience: int = 10
    subtype_lambda: float = 0.3
    seed: int = 42
    use_class_weights: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    hpo: Optional[HpoConfig] = None
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)

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
        data: Data settings.  ``data.pipeline`` must point to a pipeline ``.py`` file.

    Optional sections:
        training: Training hyperparameters (all fields have defaults).
        output: Output directory and checkpointing.
    """

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
