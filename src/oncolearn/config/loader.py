"""
YAML loading, saving, and validation for OncoLearnConfig.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Union

import yaml

from .schema import (
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
    if not config.modalities:
        raise ValueError(
            "Config must include at least one entry under 'modalities'."
        )
    if not config.model.name:
        raise ValueError("'model.name' must be a non-empty string.")

    names = [m.name for m in config.modalities]
    if len(names) != len(set(names)):
        seen, duplicates = set(), set()
        for n in names:
            (duplicates if n in seen else seen).add(n)
        raise ValueError(f"Duplicate modality names: {sorted(duplicates)}")


def load_config(path: Union[str, Path]) -> OncoLearnConfig:
    """Load an OncoLearn experiment config from a YAML file.

    The YAML must contain at minimum a ``model`` section (with a ``name`` key)
    and a ``modalities`` list (each entry must have a ``name`` key).  All other
    sections are optional and fall back to their dataclass defaults.

    Example YAML structure::

        model:
          name: gated_late_fusion
          num_stage_classes: 5

        modalities:
          - name: tabular
            cohort_code: TCGA-BRCA
            features_files: [TCGA-BRCA.mirna.tsv, pam50.tsv]

        training:
          max_epochs: 50
          learning_rate: 0.0001

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        A validated :class:`OncoLearnConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If required top-level sections are absent.
        ValueError: If validation fails (no modalities, duplicate names, etc.).
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
    if "modalities" not in raw:
        raise KeyError(f"Config '{path.name}' must contain a 'modalities' section.")

    # --- model ---
    model_raw = raw["model"]

    # Parse encoders before creating ModelConfig (they need custom handling).
    encoder_cfgs: list[EncoderConfig] = []
    for entry in model_raw.get("encoders", []):
        if "name" not in entry:
            raise KeyError(
                f"Every entry in 'model.encoders' must have a 'name' key. Got: {entry}"
            )
        enc_name = entry["name"]
        enc_output_dim = entry.get("output_dim", 128)
        enc_kwargs = {k: v for k, v in entry.items() if k not in ("name", "output_dim")}
        encoder_cfgs.append(
            EncoderConfig(name=enc_name, output_dim=enc_output_dim, kwargs=enc_kwargs)
        )

    model_cfg = _dataclass_from_dict(
        ModelConfig, {k: v for k, v in model_raw.items() if k != "encoders"}
    )
    model_cfg.encoders = encoder_cfgs

    # --- modalities ---
    # Each YAML entry is a flat dict: {"name": "tabular", <kwargs...>}
    modality_cfgs: list[ModalityConfig] = []
    for entry in raw["modalities"]:
        if "name" not in entry:
            raise KeyError(
                f"Every entry in 'modalities' must have a 'name' key. Got: {entry}"
            )
        name = entry["name"]
        kwargs = {k: v for k, v in entry.items() if k != "name"}
        modality_cfgs.append(ModalityConfig(name=name, kwargs=kwargs))

    # --- training (optional) ---
    training_cfg = _dataclass_from_dict(TrainingConfig, raw.get("training", {}))

    # --- output (optional) ---
    output_cfg = _dataclass_from_dict(OutputConfig, raw.get("output", {}))

    config = OncoLearnConfig(
        model=model_cfg,
        modalities=modality_cfgs,
        training=training_cfg,
        output=output_cfg,
        join_on=raw.get("join_on", "patient_id"),
        join_strategy=raw.get("join_strategy", "inner"),
        splits_dir=raw.get("splits_dir", None),
    )

    _validate(config)
    return config


def save_config(config: OncoLearnConfig, path: Union[str, Path]) -> None:
    """Serialize an :class:`OncoLearnConfig` to a YAML file.

    Modality ``kwargs`` are inlined as flat keys alongside ``name`` so the
    output round-trips cleanly through :func:`load_config`.

    Args:
        config: Config to serialize.
        path: Destination ``.yaml`` path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict = {}

    model_dict = dataclasses.asdict(config.model)
    # Inline encoder kwargs (same flat style as modalities).
    model_dict["encoders"] = [
        {"name": ec.name, "output_dim": ec.output_dim, **ec.kwargs}
        for ec in config.model.encoders
    ]
    raw["model"] = model_dict

    # Flatten each ModalityConfig: {name: ..., **kwargs}
    raw["modalities"] = [
        {"name": m.name, **m.kwargs} for m in config.modalities
    ]

    raw["training"] = dataclasses.asdict(config.training)
    raw["output"] = dataclasses.asdict(config.output)

    raw["join_on"] = config.join_on
    raw["join_strategy"] = config.join_strategy
    if config.splits_dir is not None:
        raw["splits_dir"] = config.splits_dir

    with path.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
