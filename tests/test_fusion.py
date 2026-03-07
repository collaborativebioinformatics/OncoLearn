"""Tests for fusion batch-key routing (Unit 3)."""

import pytest
import torch

from oncolearn.config.schema import (
    DataConfig,
    EncoderConfig,
    ModalityConfig,
    ModelConfig,
    OncoLearnConfig,
    TrainingConfig,
    OutputConfig,
)


def _make_config(enc_name: str, enc_modality=None) -> OncoLearnConfig:
    """Build a minimal OncoLearnConfig with one encoder."""
    return OncoLearnConfig(
        model=ModelConfig(
            name="gated_late_fusion",
            num_stage_classes=2,
            encoders=[
                EncoderConfig(
                    name=enc_name,
                    modality=enc_modality,
                    output_dim=64,
                )
            ],
        ),
        data=DataConfig(
            modalities=[
                ModalityConfig(
                    name=enc_modality or enc_name,
                    files=["dummy.tsv"],
                )
            ],
        ),
        training=TrainingConfig(),
        output=OutputConfig(),
    )


# ---------------------------------------------------------------------------
# Batch-key routing
# ---------------------------------------------------------------------------

def test_batch_key_uses_modality_when_set():
    """When enc_cfg.modality is set, the fusion module routes via modality name."""
    import oncolearn.modeling  # noqa: F401

    from oncolearn.modeling.modules.fusion import GatedLateFusionModule, _safe_key

    cfg = _make_config("gene", enc_modality="oncolearn.modality.gene")
    module = GatedLateFusionModule(cfg)

    # _encoder_names holds the original dotted batch key
    assert "oncolearn.modality.gene" in module._encoder_names
    # encoders ModuleDict uses the sanitized key
    assert _safe_key("oncolearn.modality.gene") in module.encoders


def test_batch_key_falls_back_to_name_when_modality_none():
    """When enc_cfg.modality is None, the encoder name is the batch key."""
    import oncolearn.modeling  # noqa: F401

    from oncolearn.modeling.modules.fusion import GatedLateFusionModule

    cfg = _make_config("gene", enc_modality=None)
    module = GatedLateFusionModule(cfg)

    assert "gene" in module._encoder_names
    assert "gene" in module.encoders


# ---------------------------------------------------------------------------
# _encode uses isinstance, not string comparison
# ---------------------------------------------------------------------------

def test_encode_dispatches_image_via_isinstance():
    """_encode checks isinstance(encoder, MRMGHierarchicalImageEncoder), not name."""
    import oncolearn.modeling  # noqa: F401
    import inspect

    from oncolearn.modeling.modules.fusion import GatedLateFusionModule
    from oncolearn.modeling.encoders.image_encoder import MRMGHierarchicalImageEncoder

    src = inspect.getsource(GatedLateFusionModule._encode)
    # Should reference isinstance, not a hardcoded string comparison
    assert "isinstance" in src
    assert 'name == "image"' not in src


# ---------------------------------------------------------------------------
# GatedLateFusionClassifier uses batch_keys from modality
# ---------------------------------------------------------------------------

def test_classifier_forward_uses_batch_keys():
    """GatedLateFusionClassifier.forward() looks up modality names in the batch."""
    import oncolearn.modeling  # noqa: F401

    from oncolearn.modeling.models.gated_late_fusion import GatedLateFusionClassifier

    cfg = _make_config("gene", enc_modality="oncolearn.modality.gene")
    clf = GatedLateFusionClassifier(cfg)

    assert clf.model._encoder_names == ["oncolearn.modality.gene"]


def test_classifier_forward_batch_key_none_modality():
    """When modality is None, batch key is the encoder name."""
    import oncolearn.modeling  # noqa: F401

    from oncolearn.modeling.models.gated_late_fusion import GatedLateFusionClassifier

    cfg = _make_config("gene", enc_modality=None)
    clf = GatedLateFusionClassifier(cfg)

    assert clf.model._encoder_names == ["gene"]
