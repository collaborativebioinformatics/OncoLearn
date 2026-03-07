from typing import Dict, List, Optional, Type, Any

# Global dictionary to store available models
_MODELS: Dict[str, Type[Any]] = {}

# Reverse mapping: model class → registry name (populated by register_model).
# Used by resolve_model_config to walk the MRO and find ancestor configs.
_CLASS_TO_NAME: Dict[type, str] = {}


def register_model(name: str, modalities: Optional[List[str]] = None):
    """
    Decorator to register an end-to-end model (e.g. PyTorch LightningModule).

    Args:
        name: Unique string name for the model (e.g., "dl_two").
        modalities: Optional list of modalities this model expects (e.g., ["image", "tabular"]).
            This is used for reference and validation when building the Trainer pipeline.
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        if name in _MODELS:
            raise ValueError(f"Model '{name}' is already registered! Cannot register {cls.__name__}.")

        # Attach expected modalities to the class for runtime checks
        cls.expected_modalities = modalities or []
        _MODELS[name] = cls
        _CLASS_TO_NAME[cls] = name
        return cls

    return wrapper


def get_model(name: str) -> Type[Any]:
    """
    Retrieve a registered model class by its string name.
    
    Args:
        name: The string name used during @register_model
        
    Returns:
        The class object of the matching model.
        
    Raises:
        KeyError if the model is not found in the registry.
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
