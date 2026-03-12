"""Registered OncoLearn data modality classes.

Importing this package triggers all @register_modality decorators.
"""
try:
    from oncolearn.data.modules.image import ImageDataModule  # noqa: F401 — triggers @register_modality("image")
except ImportError:
    pass  # pytorch_lightning not available (e.g. host/test environment)
