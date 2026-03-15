"""Shared fixtures for the OncoLearn test suite."""

import pytest

# Trigger all real @register_* decorators before snapshotting the registry.
# These imports require torch; skip gracefully when running pure-Python tests.
try:
    import oncolearn.modeling          # noqa: F401 — registers encoders + models
    import oncolearn.data.modalities   # noqa: F401 — registers modalities
    from oncolearn.registry.encoders import _ENCODERS, _CLASS_TO_NAME as _ENC_CLASS_TO_NAME
    from oncolearn.registry.modalities import _MODALITIES
    from oncolearn.registry.models import _MODELS, _CLASS_TO_NAME as _MDL_CLASS_TO_NAME
    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False


@pytest.fixture(autouse=True)
def clean_registries():
    """Snapshot and restore all registries before/after each test.

    This prevents tests that register new names from polluting other tests,
    while still allowing the real application registrations (loaded at import
    time) to remain visible when a test explicitly imports the modules.
    Skipped gracefully when torch is unavailable.
    """
    if not _HAS_REGISTRY:
        yield
        return

    enc_snap = _ENCODERS.copy()
    enc_cls_snap = _ENC_CLASS_TO_NAME.copy()
    mod_snap = _MODALITIES.copy()
    mdl_snap = _MODELS.copy()
    mdl_cls_snap = _MDL_CLASS_TO_NAME.copy()

    yield

    _ENCODERS.clear()
    _ENCODERS.update(enc_snap)
    _ENC_CLASS_TO_NAME.clear()
    _ENC_CLASS_TO_NAME.update(enc_cls_snap)

    _MODALITIES.clear()
    _MODALITIES.update(mod_snap)

    _MODELS.clear()
    _MODELS.update(mdl_snap)
    _MDL_CLASS_TO_NAME.clear()
    _MDL_CLASS_TO_NAME.update(mdl_cls_snap)


MINIMAL_TABULAR_YAML = """\
model:
  name: oncolearn.model.multimodal.gated_late_fusion
  num_stage_classes: 4
  encoders:
    - name: oncolearn.encoder.multimodal.RNABERTEncoder
      modality: oncolearn.modality.gene
      output_dim: 128

data:
  pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_xenabrowser.py
"""
