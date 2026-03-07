"""Shared fixtures for the OncoLearn test suite."""

import pytest

# Trigger all real @register_* decorators before snapshotting the registry.
# This ensures tests that query real dotted names will find them even after
# the clean_registries fixture restores a per-test snapshot.
import oncolearn.modeling          # noqa: F401 — registers encoders + models
import oncolearn.data.modalities   # noqa: F401 — registers modalities

from oncolearn.registry.encoders import _ENCODERS, _CLASS_TO_NAME as _ENC_CLASS_TO_NAME
from oncolearn.registry.modalities import _MODALITIES
from oncolearn.registry.models import _MODELS, _CLASS_TO_NAME as _MDL_CLASS_TO_NAME


@pytest.fixture(autouse=True)
def clean_registries():
    """Snapshot and restore all registries before/after each test.

    This prevents tests that register new names from polluting other tests,
    while still allowing the real application registrations (loaded at import
    time) to remain visible when a test explicitly imports the modules.
    """
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
  num_stage_classes: 5
  encoders:
    - name: oncolearn.encoder.multimodal.RNABERTEncoder
      modality: oncolearn.modality.gene
      output_dim: 128

data:
  base_directory: data/xenabrowser
  cohort_code: TCGA-BRCA
  modalities:
    - name: oncolearn.modality.gene
      join_on: patient_id
      join_strategy: inner
      files:
        - TCGA-BRCA.mirna.tsv
        - pam50.tsv
"""
