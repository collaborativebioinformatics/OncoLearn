"""Registered OncoLearn data modality classes.

Importing this package triggers all @register_modality decorators.
"""
from . import tabular  # noqa: F401 — triggers @register_modality("gene") and @register_modality("clinical")
from . import image    # noqa: F401 — triggers @register_modality("image")
