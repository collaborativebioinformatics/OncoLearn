from typing import Dict, Type, Any

_ENCODERS: Dict[str, Type[Any]] = {}

# Reverse mapping: encoder class → registry name (populated by register_encoder).
# Used by resolve_encoder_config to walk the MRO and find ancestor configs.
_CLASS_TO_NAME: Dict[type, str] = {}


def register_encoder(*names: str):
    """Decorator to register an encoder class under one or more names.

    Args:
        names: One or more string names for the encoder.  The first name is
               stored as the canonical name in ``_CLASS_TO_NAME``.
               Re-registering the *same* class under the same name is a no-op;
               re-registering a *different* class under an existing name raises
               ``ValueError``.
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        for name in names:
            if name in _ENCODERS and _ENCODERS[name] is not cls:
                raise ValueError(
                    f"Encoder '{name}' is already registered with a different class "
                    f"({_ENCODERS[name].__name__}). Cannot also register {cls.__name__}."
                )
            _ENCODERS[name] = cls
        if cls not in _CLASS_TO_NAME:
            _CLASS_TO_NAME[cls] = names[0]
        return cls

    return wrapper


def get_encoder(name: str) -> Type[Any]:
    """Retrieve a registered encoder class by its string name.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in _ENCODERS:
        raise KeyError(
            f"Encoder '{name}' not found in registry. "
            f"Available encoders: {list(_ENCODERS.keys())}"
        )
    return _ENCODERS[name]


def get_all_encoders() -> Dict[str, Type[Any]]:
    """Return a copy of the full encoder registry."""
    return _ENCODERS.copy()
