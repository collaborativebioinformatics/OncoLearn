"""Tests for the multi-name registry (Unit 1)."""

import pytest

from oncolearn.registry.encoders import register_encoder, get_encoder, get_all_encoders
from oncolearn.registry.modalities import register_modality, get_modality, get_all_modalities
from oncolearn.registry.models import register_model, get_model, get_all_models
from oncolearn.registry.datasets import register_dataset, get_dataset, get_all_datasets


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

def test_register_encoder_short_name():
    @register_encoder("test_enc")
    class MyEncoder:
        pass

    assert get_encoder("test_enc") is MyEncoder


def test_register_encoder_dotted_name():
    @register_encoder("test_enc2", "oncolearn.encoder.test.MyEncoder2")
    class MyEncoder2:
        pass

    assert get_encoder("test_enc2") is MyEncoder2
    assert get_encoder("oncolearn.encoder.test.MyEncoder2") is MyEncoder2


def test_register_encoder_both_names_same_class():
    @register_encoder("short_e", "long.dotted.e")
    class E:
        pass

    assert get_encoder("short_e") is E
    assert get_encoder("long.dotted.e") is E


def test_register_encoder_duplicate_different_class_raises():
    @register_encoder("conflict_enc")
    class A:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_encoder("conflict_enc")
        class B:
            pass


def test_register_encoder_same_class_idempotent():
    @register_encoder("idem_enc")
    class C:
        pass

    # Re-registering the same class under the same name should not raise.
    register_encoder("idem_enc")(C)
    assert get_encoder("idem_enc") is C


def test_get_unknown_encoder_raises():
    with pytest.raises(KeyError, match="not found in registry"):
        get_encoder("nonexistent_encoder_xyz")


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------

def test_register_modality_short_name():
    @register_modality("test_mod")
    class MyMod:
        pass

    assert get_modality("test_mod") is MyMod


def test_register_modality_dotted_name():
    @register_modality("test_mod3", "oncolearn.modality.test.MyMod3")
    class MyMod3:
        pass

    assert get_modality("test_mod3") is MyMod3
    assert get_modality("oncolearn.modality.test.MyMod3") is MyMod3


def test_register_modality_duplicate_different_class_raises():
    @register_modality("conflict_mod")
    class D:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_modality("conflict_mod")
        class E:
            pass


def test_get_unknown_modality_raises():
    with pytest.raises(KeyError, match="not found in registry"):
        get_modality("nonexistent_modality_xyz")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def test_register_model_short_name():
    @register_model("test_model", modalities=["image"])
    class TestModel:
        pass

    assert get_model("test_model") is TestModel
    assert TestModel.expected_modalities == ["image"]


def test_register_model_dotted_name():
    @register_model("test_model4", "oncolearn.model.test.TestModel4", modalities=["gene"])
    class TestModel4:
        pass

    assert get_model("test_model4") is TestModel4
    assert get_model("oncolearn.model.test.TestModel4") is TestModel4


def test_register_model_duplicate_different_class_raises():
    @register_model("conflict_model")
    class F:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_model("conflict_model")
        class G:
            pass


def test_get_unknown_model_raises():
    with pytest.raises(KeyError, match="not found in registry"):
        get_model("nonexistent_model_xyz")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def test_register_dataset_short_name():
    @register_dataset("test_ds")
    class MyDataset:
        pass

    assert get_dataset("test_ds") is MyDataset


def test_register_dataset_dotted_name():
    @register_dataset("test_ds2", "oncolearn.datasets.test.MyDataset2")
    class MyDataset2:
        pass

    assert get_dataset("test_ds2") is MyDataset2
    assert get_dataset("oncolearn.datasets.test.MyDataset2") is MyDataset2


def test_register_dataset_duplicate_different_class_raises():
    @register_dataset("conflict_ds")
    class H:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_dataset("conflict_ds")
        class I:
            pass


def test_register_dataset_same_class_idempotent():
    @register_dataset("idem_ds")
    class J:
        pass

    register_dataset("idem_ds")(J)
    assert get_dataset("idem_ds") is J


def test_get_unknown_dataset_raises():
    with pytest.raises(KeyError, match="not found in registry"):
        get_dataset("nonexistent_dataset_xyz")


# ---------------------------------------------------------------------------
# Real application dotted names (loaded via oncolearn.modeling import)
# ---------------------------------------------------------------------------

def test_real_dotted_encoder_names():
    """Dotted names registered by the application resolve to the same class."""
    pytest.importorskip("torch", reason="requires torch")
    import oncolearn.modeling  # noqa: F401 — triggers registration

    from oncolearn.registry.encoders import get_encoder
    gene_short = get_encoder("gene")
    gene_long = get_encoder("oncolearn.encoder.multimodal.RNABERTEncoder")
    assert gene_short is gene_long

    clinical_short = get_encoder("clinical")
    clinical_long = get_encoder("oncolearn.encoder.multimodal.ClinicalMLPEncoder")
    assert clinical_short is clinical_long

    image_short = get_encoder("image")
    image_long = get_encoder("oncolearn.encoder.multimodal.FMBCMRIEncoder")
    assert image_short is image_long


def test_real_dotted_modality_names():
    """Dotted names registered by the application resolve to the same class.

    Only the image modality is registry-based; tabular modalities (gene,
    clinical) are now loaded via the pipeline DSL and are not registered.
    """
    pytest.importorskip("pytorch_lightning", reason="requires pytorch_lightning")
    import oncolearn.data.modalities  # noqa: F401 — triggers registration

    from oncolearn.registry.modalities import get_modality
    image_short = get_modality("image")
    image_long = get_modality("oncolearn.modality.image")
    assert image_short is image_long


def test_real_dotted_model_names():
    """Dotted names registered by the application resolve to the same class."""
    pytest.importorskip("torch", reason="requires torch")
    import oncolearn.modeling  # noqa: F401

    from oncolearn.registry.models import get_model
    short = get_model("gated_late_fusion")
    long = get_model("oncolearn.model.multimodal.gated_late_fusion")
    assert short is long


def test_real_dataset_registry():
    """MultimodalDataModule is registered under its dotted name."""
    pytest.importorskip("pytorch_lightning", reason="requires pytorch_lightning")
    import oncolearn.data.modules  # noqa: F401 — triggers @register_dataset

    from oncolearn.registry.datasets import get_dataset
    from oncolearn.data.modules.multimodal import MultimodalDataModule
    assert get_dataset("oncolearn.datasets.multimodal") is MultimodalDataModule
