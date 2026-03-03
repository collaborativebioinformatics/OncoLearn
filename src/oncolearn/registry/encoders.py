from typing import Dict, Type, Any

_ENCODERS: Dict[str, Type[Any]] = {}


def register_encoder(name: str):
    """Decorator to register an encoder class by name.

    Args:
        name: Unique string name for the encoder (e.g., ``"gene"``, ``"image"``).
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        if name in _ENCODERS:
            raise ValueError(
                f"Encoder '{name}' is already registered! Cannot register {cls.__name__}."
            )
        _ENCODERS[name] = cls
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
