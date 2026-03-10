"""
YAML loading, saving, and validation for OncoLearnConfig.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Union

import yaml

from .schema import (
    DataConfig,
    EncoderConfig,
    HpoConfig,
    HpoParamSpec,
    LossConfig,
    ModalityConfig,
    ModelConfig,
    OncoLearnConfig,
    OptimizerConfig,
    OutputConfig,
    RegularizationConfig,
    SchedulerConfig,
    TrainingConfig,
)


def _dataclass_from_dict(cls, d: dict):
    """Construct a dataclass from a dict, silently ignoring unknown keys."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def _validate(config: OncoLearnConfig) -> None:
    if not config.data.modalities:
        raise ValueError(
            "Config must include at least one entry under 'data.modalities'."
        )
    if not config.model.name:
        raise ValueError("'model.name' must be a non-empty string.")

    names = [m.name for m in config.data.modalities]
    counts = Counter(names)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    if duplicates:
        raise ValueError(f"Duplicate modality names: {duplicates}")

    # Validate encoder.modality references
    for enc in config.model.encoders:
        if enc.modality is not None and enc.modality not in names:
            raise ValueError(
                f"Encoder '{enc.name}' references modality '{enc.modality}' "
                f"which is not in data.modalities. Available: {names}"
            )


def _parse_modality_entry(entry: dict) -> ModalityConfig:
    """Parse a single modality YAML entry into a ModalityConfig."""
    if "name" not in entry:
        raise KeyError(
            f"Every entry in 'data.modalities' must have a 'name' key. Got: {entry}"
        )
    name = entry["name"]
    join_on = entry.get("join_on", "patient_id")
    join_strategy = entry.get("join_strategy", "inner")
    files = entry.get("files", None)
    # Everything else is modality-specific kwargs
    reserved = {"name", "join_on", "join_strategy", "files"}
    kwargs = {k: v for k, v in entry.items() if k not in reserved}
    return ModalityConfig(
        name=name,
        join_on=join_on,
        join_strategy=join_strategy,
        files=files,
        kwargs=kwargs,
    )


def _parse_data_section(raw: dict) -> DataConfig:
    """Parse the ``data:`` section of a YAML config."""
    data_raw = raw.get("data", {})
    modality_entries = data_raw.get("modalities", [])
    modality_cfgs = [_parse_modality_entry(e) for e in modality_entries]
    return DataConfig(
        modalities=modality_cfgs,
        base_directory=data_raw.get("base_directory", "data/xenabrowser"),
        cohort_code=data_raw.get("cohort_code", "TCGA-BRCA"),
        splits_dir=data_raw.get("splits_dir", None),
    )



def _parse_training_section(raw: dict) -> TrainingConfig:
    """Parse the ``training:`` section, handling nested optimizer/scheduler/loss/regularization."""
    nested_keys = {"optimizer", "scheduler", "loss", "regularization"}
    flat = {k: v for k, v in raw.items() if k not in nested_keys}
    training_cfg = _dataclass_from_dict(TrainingConfig, flat)

    if "optimizer" in raw:
        opt_raw = raw["optimizer"]
        training_cfg.optimizer = OptimizerConfig(
            name=opt_raw.get("name", "torch.optim.AdamW"),
            params=opt_raw.get("params", {}),
        )

    if "scheduler" in raw:
        sched_raw = raw["scheduler"]
        training_cfg.scheduler = SchedulerConfig(
            name=sched_raw.get("name", "torch.optim.lr_scheduler.CosineAnnealingLR"),
            params=sched_raw.get("params", {}),
            monitor=sched_raw.get("monitor", "val_loss"),
            interval=sched_raw.get("interval", "epoch"),
            frequency=sched_raw.get("frequency", 1),
        )

    if "loss" in raw:
        loss_raw = raw["loss"]
        training_cfg.loss = LossConfig(
            name=loss_raw.get("name", "torch.nn.CrossEntropyLoss"),
            params=loss_raw.get("params", {}),
        )

    if "regularization" in raw:
        reg_raw = raw["regularization"]
        training_cfg.regularization = RegularizationConfig(
            l1_lambda=reg_raw.get("l1_lambda", 0.0),
            gradient_clip_val=reg_raw.get("gradient_clip_val", 0.0),
            label_smoothing=reg_raw.get("label_smoothing", 0.0),
        )
        
    if "hpo" in raw: 
        training_cfg.hpo = _parse_hpo_section(raw["hpo"])   
    
    return training_cfg


def _parse_hpo_section(raw: dict) -> HpoConfig:
    """Parse the ``training.hpo:`` section into an :class:`HpoConfig`."""
    search_space: dict[str, HpoParamSpec] = {}
    optimizers: dict[str, dict[str, HpoParamSpec]] = {}
    losses: dict[str, dict[str, HpoParamSpec]] = {}

    for path, spec_raw in raw.get("search_space", {}).items():
        if path == "training.optimizers":
            optimizers = _parse_conditional_params(spec_raw, "training.optimizers")
        elif path == "training.losses":
            losses = _parse_conditional_params(spec_raw, "training.losses")
        else:
            if not isinstance(spec_raw, dict) or "type" not in spec_raw:
                raise ValueError(
                    f"HPO search_space entry '{path}' must be a dict with a 'type' key."
                )
            search_space[path] = HpoParamSpec(
                type=spec_raw["type"],
                low=spec_raw.get("low"),
                high=spec_raw.get("high"),
                log=spec_raw.get("log", False),
                step=spec_raw.get("step"),
                choices=spec_raw.get("choices"),
            )

    return HpoConfig(
        n_trials=raw.get("n_trials", 20),
        study_name=raw.get("study_name", "oncolearn_hpo"),
        storage=raw.get("storage"),
        direction=raw.get("direction", "maximize"),
        metric=raw.get("metric", "val_acc"),
        pruning=raw.get("pruning", True),
        epochs_per_trial=raw.get("epochs_per_trial"),
        seed=raw.get("seed", 42),
        search_space=search_space,
        optimizers=optimizers,
        losses=losses,
    )


def _parse_conditional_params(
    raw: dict, section_name: str
) -> dict[str, dict[str, HpoParamSpec]]:
    """Parse a ``{class_name: {param: spec}}`` nested block into HpoParamSpec dicts."""
    result: dict[str, dict[str, HpoParamSpec]] = {}
    for class_name, param_specs in raw.items():
        result[class_name] = {}
        for param_name, spec_raw in param_specs.items():
            if not isinstance(spec_raw, dict) or "type" not in spec_raw:
                raise ValueError(
                    f"HPO {section_name}['{class_name}']['{param_name}'] "
                    "must be a dict with a 'type' key."
                )
            result[class_name][param_name] = HpoParamSpec(
                type=spec_raw["type"],
                low=spec_raw.get("low"),
                high=spec_raw.get("high"),
                log=spec_raw.get("log", False),
                step=spec_raw.get("step"),
                choices=spec_raw.get("choices"),
            )
    return result


def load_config(path: Union[str, Path]) -> OncoLearnConfig:
    """Load an OncoLearn experiment config from a YAML file.

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        A validated :class:`OncoLearnConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If required top-level sections are absent.
        ValueError: If validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Config file is empty or invalid YAML: {path}")

    if "model" not in raw:
        raise KeyError(f"Config '{path.name}' must contain a 'model' section.")

    # --- model ---
    model_raw = raw["model"]

    encoder_cfgs: list[EncoderConfig] = []
    for entry in model_raw.get("encoders", []):
        if "name" not in entry:
            raise KeyError(
                f"Every entry in 'model.encoders' must have a 'name' key. Got: {entry}"
            )
        enc_name = entry["name"]
        enc_modality = entry.get("modality", None)
        enc_output_dim = entry.get("output_dim", 128)
        reserved = {"name", "modality", "output_dim"}
        enc_kwargs = {k: v for k, v in entry.items() if k not in reserved}
        encoder_cfgs.append(
            EncoderConfig(
                name=enc_name,
                modality=enc_modality,
                output_dim=enc_output_dim,
                kwargs=enc_kwargs,
            )
        )

    model_cfg = _dataclass_from_dict(
        ModelConfig, {k: v for k, v in model_raw.items() if k != "encoders"}
    )
    model_cfg.encoders = encoder_cfgs

    # --- data section ---
    if "data" not in raw:
        raise KeyError(
            f"Config '{path.name}' must contain a 'data' section with a 'modalities' list."
        )
    if not raw["data"].get("modalities"):
        raise KeyError(
            f"Config '{path.name}': 'data' section must contain a 'modalities' list."
        )
    data_cfg = _parse_data_section(raw)

    # --- training (optional) ---
    training_cfg = _parse_training_section(raw.get("training", {}))

    # --- output (optional) ---
    output_cfg = _dataclass_from_dict(OutputConfig, raw.get("output", {}))

    config = OncoLearnConfig(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        output=output_cfg,
    )

    _validate(config)
    return config


def save_config(config: OncoLearnConfig, path: Union[str, Path]) -> None:
    """Serialize an :class:`OncoLearnConfig` to a YAML file.

    Uses the new ``data:`` section format.  Modality ``kwargs`` are inlined as
    flat keys alongside the first-class fields so the output round-trips
    cleanly through :func:`load_config`.

    Args:
        config: Config to serialize.
        path: Destination ``.yaml`` path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict = {}

    # --- model ---
    model_dict = dataclasses.asdict(config.model)
    model_dict["encoders"] = []
    for ec in config.model.encoders:
        entry = {"name": ec.name, "output_dim": ec.output_dim, **ec.kwargs}
        if ec.modality is not None:
            entry["modality"] = ec.modality
        model_dict["encoders"].append(entry)
    raw["model"] = model_dict

    # --- data ---
    data_dict: dict = {
        "base_directory": config.data.base_directory,
        "cohort_code": config.data.cohort_code,
    }
    if config.data.splits_dir is not None:
        data_dict["splits_dir"] = config.data.splits_dir

    data_dict["modalities"] = []
    for m in config.data.modalities:
        entry = {
            "name": m.name,
            "join_on": m.join_on,
            "join_strategy": m.join_strategy,
        }
        if m.files is not None:
            entry["files"] = m.files
        entry.update(m.kwargs)
        data_dict["modalities"].append(entry)
    raw["data"] = data_dict

    t = config.training
    raw["training"] = {
        "max_epochs": t.max_epochs,
        "batch_size": t.batch_size,
        "num_workers": t.num_workers,
        "accelerator": t.accelerator,
        "devices": t.devices,
        "early_stopping_patience": t.early_stopping_patience,
        "subtype_lambda": t.subtype_lambda,
        "seed": t.seed,
        "use_class_weights": t.use_class_weights,
        "optimizer": {"name": t.optimizer.name, "params": dict(t.optimizer.params)},
        "scheduler": {
            "name": t.scheduler.name,
            "params": dict(t.scheduler.params),
            "monitor": t.scheduler.monitor,
            "interval": t.scheduler.interval,
            "frequency": t.scheduler.frequency,
        },
        "loss": {"name": t.loss.name, "params": dict(t.loss.params)},
        "regularization": {
            "l1_lambda": t.regularization.l1_lambda,
            "gradient_clip_val": t.regularization.gradient_clip_val,
            "label_smoothing": t.regularization.label_smoothing,
        },
    }
    raw["output"] = dataclasses.asdict(config.output)

    with path.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
