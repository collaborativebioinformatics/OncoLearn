from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path


class BaseDataLoader(ABC):
    """
    Abstract Base Class for all data loaders (image and tabular).
    Defines the protocol: check if a file can be handled, then load it.
    """

    @classmethod
    @abstractmethod
    def can_load(cls, file_path: Path) -> bool:
        """Return True if this loader can process the given file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: Path) -> Any:
        """Load the file at *file_path* and return a normalized representation."""
        pass

