"""Registered OncoLearn data modality classes.

Importing this package triggers all @register_modality decorators.
"""
from . import tabular  # noqa: F401 — triggers @register_modality("tabular")
from . import image    # noqa: F401 — triggers @register_modality("image")
