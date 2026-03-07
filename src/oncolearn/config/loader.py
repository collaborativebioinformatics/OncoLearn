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
    ModalityConfig,
    ModelConfig,
    OncoLearnConfig,
    OutputConfig,
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


def _parse_legacy_modalities(raw: dict) -> DataConfig:
    """Convert old top-level ``modalities:`` list to ``DataConfig`` (backward compat)."""
    modality_cfgs = []
    for entry in raw.get("modalities", []):
        if "name" not in entry:
            raise KeyError(
                f"Every entry in 'modalities' must have a 'name' key. Got: {entry}"
            )
        name = entry["name"]
        # Legacy format: all non-name keys are kwargs; no first-class join_on/files
        kwargs = {k: v for k, v in entry.items() if k != "name"}
        # Pull out features_files → files if present
        files = kwargs.pop("features_files", None)
        modality_cfgs.append(ModalityConfig(
            name=name,
            join_on=raw.get("join_on", "patient_id"),
            join_strategy=raw.get("join_strategy", "inner"),
            files=files,
            kwargs=kwargs,
        ))
    return DataConfig(
        modalities=modality_cfgs,
        base_directory="data/xenabrowser",
        cohort_code="TCGA-BRCA",
        splits_dir=raw.get("splits_dir", None),
    )


def load_config(path: Union[str, Path]) -> OncoLearnConfig:
    """Load an OncoLearn experiment config from a YAML file.

    Supports both the new ``data:`` section format and the legacy top-level
    ``modalities:`` list format.

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

    # --- data section (new format) or legacy modalities list ---
    if "data" in raw:
        if not raw["data"].get("modalities"):
            raise KeyError(
                f"Config '{path.name}': 'data' section must contain a 'modalities' list."
            )
        data_cfg = _parse_data_section(raw)
    elif "modalities" in raw:
        data_cfg = _parse_legacy_modalities(raw)
    else:
        raise KeyError(
            f"Config '{path.name}' must contain either a 'data' section or a "
            "'modalities' list."
        )

    # --- training (optional) ---
    training_cfg = _dataclass_from_dict(TrainingConfig, raw.get("training", {}))

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

    raw["training"] = dataclasses.asdict(config.training)
    raw["output"] = dataclasses.asdict(config.output)

    with path.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
