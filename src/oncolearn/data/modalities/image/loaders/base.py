from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

class BaseImageLoader(ABC):
    """
    Abstract Base Class for all Image Loaders.
    Guarantees an extensible protocol for `ImageDataModule`.
    """
    
    @classmethod
    @abstractmethod
    def can_load(cls, file_path: Path) -> bool:
        """
        Returns True if this loader can process the given file extension.
        """
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, file_path: Path) -> Any:
        """
        Loads the image at the specified path and returns a normalized representation
        (e.g., a PIL Image or a PyTorch tensor).
        """
        pass
