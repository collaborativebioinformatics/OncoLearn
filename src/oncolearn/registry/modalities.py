from typing import Dict, Type, Any

# Global dictionary to store available modalities (e.g., DataModules)
_MODALITIES: Dict[str, Type[Any]] = {}


def register_modality(*names: str):
    """Decorator to register a modality data module under one or more names.

    Args:
        names: One or more string names for the modality.
               Re-registering the *same* class under the same name is a no-op;
               re-registering a *different* class under an existing name raises
               ``ValueError``.
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        for name in names:
            if name in _MODALITIES and _MODALITIES[name] is not cls:
                raise ValueError(
                    f"Modality '{name}' is already registered with a different class "
                    f"({_MODALITIES[name].__name__}). Cannot also register {cls.__name__}."
                )
            _MODALITIES[name] = cls
        return cls

    return wrapper


def get_modality(name: str) -> Type[Any]:
    """Retrieve a registered modality class by its string name.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in _MODALITIES:
        raise KeyError(
            f"Modality '{name}' not found in registry. "
            f"Available modalities: {list(_MODALITIES.keys())}"
        )
    return _MODALITIES[name]


def get_all_modalities() -> Dict[str, Type[Any]]:
    """Return a dictionary of all registered modalities."""
    return _MODALITIES.copy()
