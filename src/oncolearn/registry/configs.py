"""
Config registry for registered encoders and models.

Usage::

    from oncolearn.registry import register_config

    @register_config("oncolearn.encoder.multimodal.RNABERTEncoder")
    @dataclass
    class RNABERTEncoderConfig:
        output_dim: int = 128
        max_seq_len: int = 512

When resolving, the MRO of the encoder class is walked top-down.  Any ancestor
that is itself a registered encoder/model with a registered config contributes
its defaults first; child configs overwrite parent fields.  Finally, the
matching ``EncoderConfig.kwargs`` (and ``output_dim``) from the
``OncoLearnConfig`` YAML override everything.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Type

# name → config class (same name as the registered encoder/model)
_CONFIGS: Dict[str, Type] = {}


def register_config(name: str):
    """Decorator that registers a config class under *name*.

    Use the same *name* as the corresponding ``@register_encoder`` or
    ``@register_model`` decorator.
    """
    def wrapper(cls: Type) -> Type:
        _CONFIGS[name] = cls
        return cls
    return wrapper


def get_config(name: str) -> Type:
    """Return the config class registered under *name*.

    Raises:
        KeyError: if no config is registered for *name*.
    """
    if name not in _CONFIGS:
        raise KeyError(
            f"No config registered for '{name}'. "
            f"Available: {list(_CONFIGS.keys())}"
        )
    return _CONFIGS[name]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_defaults(cfg_cls: Type) -> Dict[str, Any]:
    """Extract field default values from a dataclass or plain config class."""
    if dataclasses.is_dataclass(cfg_cls):
        result: Dict[str, Any] = {}
        for f in dataclasses.fields(cfg_cls):
            if f.default is not dataclasses.MISSING:
                result[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                result[f.name] = f.default_factory()
        return result
    # Plain class: collect class-level non-callable, non-dunder attributes.
    return {
        k: v for k, v in vars(cfg_cls).items()
        if not k.startswith("_")
        and not callable(v)
        and not isinstance(v, (classmethod, staticmethod, property))
    }


def _instantiate(cfg_cls: Type, kwargs: Dict[str, Any]) -> Any:
    """Create an instance of *cfg_cls* using only the keys it accepts."""
    if dataclasses.is_dataclass(cfg_cls):
        valid = {f.name for f in dataclasses.fields(cfg_cls)}
    else:
        valid = set(_get_defaults(cfg_cls).keys())

    filtered = {k: v for k, v in kwargs.items() if k in valid}

    if dataclasses.is_dataclass(cfg_cls):
        return cfg_cls(**filtered)

    instance = object.__new__(cfg_cls)
    for k, v in filtered.items():
        setattr(instance, k, v)
    return instance


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_encoder_config(encoder_cls: type, onco_config: Any) -> Any:
    """Resolve the encoder config instance for *encoder_cls*.

    Algorithm:
    1. Walk ``encoder_cls.__mro__`` from top (most-base) to bottom (leaf).
    2. For each ancestor that is a registered encoder with a registered config,
       collect that config class.
    3. Merge defaults parent-first so child fields overwrite parent fields.
    4. Apply overrides from the matching ``onco_config.model.encoders`` entry
       (matched by class identity): ``output_dim`` and all ``kwargs`` entries.
    5. Instantiate (and return) the leaf registered config class, passing only
       the fields it declares.

    If no config class is registered for *encoder_cls*, returns a
    ``types.SimpleNamespace`` built from the merged dict.
    """
    from .encoders import _CLASS_TO_NAME, _ENCODERS

    cfg_chain: list[Type] = []
    for cls in reversed(encoder_cls.__mro__):
        name = _CLASS_TO_NAME.get(cls)
        if name and name in _CONFIGS:
            cfg_chain.append(_CONFIGS[name])

    merged: Dict[str, Any] = {}
    for cfg_cls in cfg_chain:
        merged.update(_get_defaults(cfg_cls))

    # Override with values from the YAML EncoderConfig entry matched by class identity.
    enc_entry = next(
        (e for e in onco_config.model.encoders if _ENCODERS.get(e.name) is encoder_cls),
        None,
    )
    if enc_entry is not None:
        merged["output_dim"] = enc_entry.output_dim
        merged.update(enc_entry.kwargs)

    # Use the most-derived config class found in the MRO walk as the leaf.
    leaf_cfg_cls = cfg_chain[-1] if cfg_chain else None
    if leaf_cfg_cls is None:
        import types
        return types.SimpleNamespace(**merged)

    return _instantiate(leaf_cfg_cls, merged)


def resolve_model_config(model_cls: type, onco_config: Any) -> Any:
    """Resolve the model config instance for *model_cls*.

    Same MRO-walk and merge strategy as :func:`resolve_encoder_config`.
    Overrides are sourced from the fields of ``onco_config.model``
    (a ``ModelConfig`` dataclass) rather than a per-encoder kwargs dict.
    """
    from .models import _CLASS_TO_NAME

    cfg_chain: list[Type] = []
    for cls in reversed(model_cls.__mro__):
        name = _CLASS_TO_NAME.get(cls)
        if name and name in _CONFIGS:
            cfg_chain.append(_CONFIGS[name])

    merged: Dict[str, Any] = {}
    for cfg_cls in cfg_chain:
        merged.update(_get_defaults(cfg_cls))

    # Override with matching fields from onco_config.model.
    if dataclasses.is_dataclass(onco_config.model):
        for f in dataclasses.fields(onco_config.model):
            if f.name in merged:
                merged[f.name] = getattr(onco_config.model, f.name)

    leaf_cfg_cls = cfg_chain[-1] if cfg_chain else None
    if leaf_cfg_cls is None:
        import types
        return types.SimpleNamespace(**merged)

    return _instantiate(leaf_cfg_cls, merged)
