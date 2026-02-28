from typing import Dict, Type, Any

# Global dictionary to store available modalities (e.g., DataModules)
_MODALITIES: Dict[str, Type[Any]] = {}


def register_modality(name: str):
    """
    Decorator to register a modality data module (e.g. LightningDataModule or Dataset).
    
    Args:
        name: Unique string name for the modality (e.g., "image", "tabular")
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        if name in _MODALITIES:
            raise ValueError(f"Modality '{name}' is already registered! Cannot register {cls.__name__}.")
            
        _MODALITIES[name] = cls
        return cls
        
    return wrapper


def get_modality(name: str) -> Type[Any]:
    """
    Retrieve a registered modality class by its string name.
    
    Args:
        name: The string name used during @register_modality
        
    Returns:
        The class object of the matching modality data module.
        
    Raises:
        KeyError if the modality is not found in the registry.
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
