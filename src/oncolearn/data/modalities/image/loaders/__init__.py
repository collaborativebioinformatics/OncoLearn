from .base import BaseImageLoader
from .dicom_loader import DicomLoader
from .pillow_loader import PillowLoader

__all__ = [
    "BaseImageLoader",
    "DicomLoader",
    "PillowLoader",
]

# Order matters: more specific loaders first
DEFAULT_LOADERS = [
    DicomLoader,
    PillowLoader
]
