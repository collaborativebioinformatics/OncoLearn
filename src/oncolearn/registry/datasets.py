from typing import Dict, Type, Any

_DATASETS: Dict[str, Type[Any]] = {}


def register_dataset(*names: str):
    """Decorator to register a dataset class under one or more names.

    Args:
        names: One or more string names for the dataset.  Re-registering the
               *same* class under the same name is a no-op; re-registering a
               *different* class under an existing name raises ``ValueError``.
    """
    def wrapper(cls: Type[Any]) -> Type[Any]:
        for name in names:
            if name in _DATASETS and _DATASETS[name] is not cls:
                raise ValueError(
                    f"Dataset '{name}' is already registered with a different class "
                    f"({_DATASETS[name].__name__}). Cannot also register {cls.__name__}."
                )
            _DATASETS[name] = cls
        return cls

    return wrapper


def get_dataset(name: str) -> Type[Any]:
    """Retrieve a registered dataset class by its string name.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in _DATASETS:
        raise KeyError(
            f"Dataset '{name}' not found in registry. "
            f"Available datasets: {list(_DATASETS.keys())}"
        )
    return _DATASETS[name]


def get_all_datasets() -> Dict[str, Type[Any]]:
    """Return a copy of the full dataset registry."""
    return _DATASETS.copy()
