from .base import BaseDataLoader
from .dicom_loader import DicomLoader
from .pillow_loader import PillowLoader
from .tabular_loader import XenabrowserParser

__all__ = [
    "BaseDataLoader",
    "DicomLoader",
    "PillowLoader",
    "XenabrowserParser",
]

# Order matters: more specific loaders first
DEFAULT_LOADERS = [
    DicomLoader,
    PillowLoader
]
