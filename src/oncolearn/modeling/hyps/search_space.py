"""
Config-driven Optuna search-space helpers.

The search space is fully defined by ``training.hpo`` in the YAML config — no
hardcoded parameter lists live here.

Flat dotted-path params
-----------------------
``hpo_cfg.search_space`` maps dotted config paths to :class:`~oncolearn.config.HpoParamSpec`.
These are sampled unconditionally on every trial.

Conditional optimizer / loss params
------------------------------------
``hpo_cfg.optimizers`` maps optimizer class names to per-param specs.  When
multiple optimizers are present, Optuna also samples a categorical choice over
which one to use.  Only the params for the *chosen* optimizer are sampled in
each trial, preventing invalid kwargs (e.g. ``momentum`` passed to AdamW).

``hpo_cfg.losses`` works identically for the loss function.

Optuna trial keys for conditional params are namespaced as
``opt.<class_name>.<param>`` / ``loss.<class_name>.<param>`` so that TPE can
build independent priors per class across trials.

:func:`suggest_hyperparams` samples all parameters for one Optuna trial and
returns a modified deep copy of the base config.
:func:`apply_params` applies a plain ``{path: value}`` dict without a trial
object — used to reconstruct the best config after a study finishes.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from oncolearn.config import HpoConfig, HpoParamSpec, OncoLearnConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_hyperparams(
    trial: optuna.Trial,
    base_config: "OncoLearnConfig",
    hpo_cfg: "HpoConfig",
) -> "OncoLearnConfig":
    """Apply Optuna trial suggestions to a deep copy of *base_config*.

    Flat params in ``hpo_cfg.search_space`` are sampled unconditionally.
    ``hpo_cfg.optimizers`` and ``hpo_cfg.losses`` are sampled conditionally —
    if multiple classes are listed, a categorical choice is sampled first, then
    only that class's params are suggested.

    Args:
        trial:       Active Optuna trial.
        base_config: Template config (not mutated).
        hpo_cfg:     Parsed :class:`~oncolearn.config.HpoConfig`.

    Returns:
        New :class:`~oncolearn.config.OncoLearnConfig` with sampled values.
    """
    config = copy.deepcopy(base_config)
    params: Dict[str, Any] = {}

    # --- flat (unconditional) search space ---
    for path, spec in hpo_cfg.search_space.items():
        params[path] = _suggest_one(trial, path, spec)

    # --- conditional optimizer params ---
    if hpo_cfg.optimizers:
        opt_names = list(hpo_cfg.optimizers.keys())
        if len(opt_names) > 1:
            chosen_opt = trial.suggest_categorical("training.optimizer.name", opt_names)
        else:
            chosen_opt = opt_names[0]
        params["training.optimizer.name"] = chosen_opt
        opt_param_values: Dict[str, Any] = {}
        for param_name, spec in hpo_cfg.optimizers[chosen_opt].items():
            trial_key = f"opt.{chosen_opt}.{param_name}"
            opt_param_values[param_name] = _suggest_one(trial, trial_key, spec)
        params["training.optimizer.params"] = opt_param_values

    # --- conditional loss params ---
    if hpo_cfg.losses:
        loss_names = list(hpo_cfg.losses.keys())
        if len(loss_names) > 1:
            chosen_loss = trial.suggest_categorical("training.loss.name", loss_names)
        else:
            chosen_loss = loss_names[0]
        params["training.loss.name"] = chosen_loss
        loss_param_values: Dict[str, Any] = {}
        for param_name, spec in hpo_cfg.losses[chosen_loss].items():
            trial_key = f"loss.{chosen_loss}.{param_name}"
            loss_param_values[param_name] = _suggest_one(trial, trial_key, spec)
        params["training.loss.params"] = loss_param_values

    apply_params(config, params)
    return config


def apply_params(config: "OncoLearnConfig", params: Dict[str, Any]) -> None:
    """Write ``{path: value}`` pairs into *config* in-place.

    Each key is a dotted config path walked via dataclass attrs, dict keys, or
    list indices.  Dict values (e.g. ``training.optimizer.params``) replace the
    target attribute wholesale.  Examples::

        "training.optimizer.name"    → config.training.optimizer.name
        "training.optimizer.params"  → config.training.optimizer.params (dict replace)
        "training.loss.params"       → config.training.loss.params (dict replace)
        "training.batch_size"        → config.training.batch_size
        "model.encoders.0.output_dim"→ config.model.encoders[0].output_dim
    """
    for path, value in params.items():
        _set_nested(config, path.split("."), value)


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
