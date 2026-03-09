"""
Config-driven Optuna search-space helpers.

The search space is fully defined by ``training.hpo.search_space`` in the YAML
config — no hardcoded parameter lists live here.  Each key is a dotted config
path (e.g. ``"training.optimizer.params.lr"``); each value is an
:class:`~oncolearn.config.HpoParamSpec` describing the sampling type and range.

Conditional optimizer params
-----------------------------
When ``training.hpo.optimizer_params`` is defined, optimizer-specific
hyperparameters (lr, momentum, weight_decay, etc.) are sampled *conditionally*:
only the params for the optimizer chosen in a given trial are suggested.  This
prevents invalid kwargs (e.g. ``momentum`` passed to AdamW) from reaching the
optimizer constructor.

The Optuna trial keys for these params are namespaced as
``opt_param.<optimizer_dotted_name>.<param_name>`` so that TPE can build
independent priors for each optimizer's parameters across trials.

:func:`suggest_hyperparams` samples all parameters for one Optuna trial and
returns a modified deep copy of the base config.
:func:`apply_params` applies a plain dict of ``{path: value}`` without a trial
object — used to reconstruct the best config after a study finishes.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from oncolearn.config import HpoConfig, HpoParamSpec, OncoLearnConfig

# Prefix used to namespace optimizer-conditional trial params.
_OPT_PARAM_PREFIX = "opt_param."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_hyperparams(
    trial: optuna.Trial,
    base_config: "OncoLearnConfig",
    hpo_cfg: "HpoConfig",
) -> "OncoLearnConfig":
    """Apply Optuna trial suggestions to a deep copy of *base_config*.

    Regular params in ``hpo_cfg.search_space`` are sampled unconditionally.
    If ``hpo_cfg.optimizer_params`` is non-empty, the optimizer-specific params
    for the chosen optimizer are sampled additionally under namespaced trial
    keys (see module docstring).

    Args:
        trial:       Active Optuna trial.
        base_config: Template config (not mutated).
        hpo_cfg:     Parsed :class:`~oncolearn.config.HpoConfig`.

    Returns:
        New :class:`~oncolearn.config.OncoLearnConfig` with sampled values.
    """
    config = copy.deepcopy(base_config)
    params: Dict[str, Any] = {}
    chosen_opt_name: Optional[str] = None

    # --- regular (unconditional) search space ---
    for path, spec in hpo_cfg.search_space.items():
        value = _suggest_one(trial, path, spec)
        params[path] = value
        if path == "training.optimizer.name":
            chosen_opt_name = value

    # --- conditional optimizer params ---
    if hpo_cfg.optimizer_params:
        if chosen_opt_name is None:
            # Optimizer name not being tuned — use the base config value.
            chosen_opt_name = base_config.training.optimizer.name
        opt_specs = hpo_cfg.optimizer_params.get(chosen_opt_name, {})
        for param_name, spec in opt_specs.items():
            trial_key = f"{_OPT_PARAM_PREFIX}{chosen_opt_name}.{param_name}"
            params[trial_key] = _suggest_one(trial, trial_key, spec)

    apply_params(config, params)
    return config


def apply_params(config: "OncoLearnConfig", params: Dict[str, Any]) -> None:
    """Write ``{path: value}`` pairs into *config* in-place.

    Handles three kinds of keys:

    * Regular dotted config paths — walked via dataclass attrs, dict keys, or
      list indices.  Examples::

          "training.optimizer.params.lr"        → config.training.optimizer.params["lr"]
          "training.batch_size"                  → config.training.batch_size
          "model.encoders.0.output_dim"          → config.model.encoders[0].output_dim

    * ``"training.optimizer.name"`` — sets the optimizer class string; must be
      applied *before* the ``opt_param.*`` keys so the chosen name is known.

    * ``"opt_param.<optimizer_name>.<param_name>"`` keys — produced by
      :func:`suggest_hyperparams` for conditional optimizer params.  Only the
      entries for the *chosen* optimizer are applied; the rest are ignored.
      When any such entries are present the entire ``optimizer.params`` dict is
      replaced so no leftover params from other optimizers remain.
    """
    opt_param_entries: Dict[str, Any] = {}
    regular_params: Dict[str, Any] = {}
    chosen_opt: Optional[str] = None

    for path, value in params.items():
        if path.startswith(_OPT_PARAM_PREFIX):
            opt_param_entries[path] = value
        else:
            regular_params[path] = value
            if path == "training.optimizer.name":
                chosen_opt = value

    # Apply regular params first (updates optimizer.name if present).
    for path, value in regular_params.items():
        _set_nested(config, path.split("."), value)

    # Apply conditional optimizer params.
    if opt_param_entries:
        if chosen_opt is None:
            # Name was set via regular_params above or left at its base value.
            chosen_opt = config.training.optimizer.name
        prefix = f"{_OPT_PARAM_PREFIX}{chosen_opt}."
        new_opt_params: Dict[str, Any] = {}
        for key, value in opt_param_entries.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                new_opt_params[param_name] = value
            # Keys for non-chosen optimizers are silently skipped.
        if new_opt_params:
            config.training.optimizer.params = new_opt_params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suggest_one(
    trial: optuna.Trial,
    name: str,
    spec: "HpoParamSpec",
) -> Any:
    """Call the appropriate ``trial.suggest_*`` method for *spec*."""
    if spec.type == "float":
        if spec.low is None or spec.high is None:
            raise ValueError(
                f"HPO param '{name}': 'low' and 'high' are required for type 'float'."
            )
        kwargs: Dict[str, Any] = {"log": spec.log}
        if spec.step is not None:
            kwargs["step"] = float(spec.step)
        return trial.suggest_float(name, float(spec.low), float(spec.high), **kwargs)

    elif spec.type == "int":
        if spec.low is None or spec.high is None:
            raise ValueError(
                f"HPO param '{name}': 'low' and 'high' are required for type 'int'."
            )
        kwargs = {}
        if spec.step is not None:
            kwargs["step"] = int(spec.step)
        return trial.suggest_int(name, int(spec.low), int(spec.high), **kwargs)

    elif spec.type == "categorical":
        if not spec.choices:
            raise ValueError(
                f"HPO param '{name}': 'choices' is required for type 'categorical'."
            )
        return trial.suggest_categorical(name, spec.choices)

    else:
        raise ValueError(
            f"HPO param '{name}': unknown type '{spec.type}'. "
            "Expected 'float', 'int', or 'categorical'."
        )


def _set_nested(obj: Any, keys: list[str], value: Any) -> None:
    """Walk *obj* along *keys* and set the leaf to *value*."""
    for key in keys[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list):
            obj = obj[int(key)]
        else:
            obj = getattr(obj, key)

    last = keys[-1]
    if isinstance(obj, dict):
        obj[last] = value
    elif isinstance(obj, list):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)
