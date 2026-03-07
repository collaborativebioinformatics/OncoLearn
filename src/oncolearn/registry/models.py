from typing import Dict, List, Optional, Type, Any

# Global dictionary to store available models
_MODELS: Dict[str, Type[Any]] = {}

# Reverse mapping: model class → registry name (populated by register_model).
_CLASS_TO_NAME: Dict[type, str] = {}


def register_model(*names: str, modalities: Optional[List[str]] = None):
    """Decorator to register an end-to-end model under one or more names.

    Args:
        names: One or more string names for the model.  The first name is
               stored as the canonical name in ``_CLASS_TO_NAME``.
               Re-registering the *same* class under the same name is a no-op;
               re-registering a *different* class under an existing name raises
               ``ValueError``.
        modalities: Optional list of modalities this model expects.
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        for name in names:
            if name in _MODELS and _MODELS[name] is not cls:
                raise ValueError(
                    f"Model '{name}' is already registered with a different class "
                    f"({_MODELS[name].__name__}). Cannot also register {cls.__name__}."
                )
            _MODELS[name] = cls

        cls.expected_modalities = modalities or []
        if cls not in _CLASS_TO_NAME:
            _CLASS_TO_NAME[cls] = names[0]
        return cls

    return wrapper


def get_model(name: str) -> Type[Any]:
    """Retrieve a registered model class by its string name.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in _MODELS:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(_MODELS.keys())}"
        )
    return _MODELS[name]


def get_all_models() -> Dict[str, Type[Any]]:
    """Return a dictionary of all registered models."""
    return _MODELS.copy()
