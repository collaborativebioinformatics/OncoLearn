import pytest
from oncolearn.registry.models import register_model, get_model, get_all_models, _MODELS
from oncolearn.registry.modalities import register_modality, get_modality, get_all_modalities, _MODALITIES


@pytest.fixture(autouse=True)
def clean_registries():
    """Clear registries before each test to ensure isolation."""
    _MODELS.clear()
    _MODALITIES.clear()
    yield


def test_register_model_success():
    @register_model("test_model", modalities=["image"])
    class TestModel:
        pass
        
    assert "test_model" in get_all_models()
    assert get_model("test_model") == TestModel
    assert TestModel.expected_modalities == ["image"]


def test_register_model_duplicate_throws_error():
    @register_model("test_model")
    class TestModel1:
        pass
        
    with pytest.raises(ValueError, match="is already registered"):
        @register_model("test_model")
        class TestModel2:
            pass


def test_get_unknown_model_throws_error():
    with pytest.raises(KeyError, match="not found in registry"):
        get_model("unknown_model")


def test_register_modality_success():
    @register_modality("test_modality")
    class TestModality:
        pass
        
    assert "test_modality" in get_all_modalities()
    assert get_modality("test_modality") == TestModality


def test_register_modality_duplicate_throws_error():
    @register_modality("test_modality")
    class TestModality1:
        pass
        
    with pytest.raises(ValueError, match="is already registered"):
        @register_modality("test_modality")
        class TestModality2:
            pass


def test_get_unknown_modality_throws_error():
    with pytest.raises(KeyError, match="not found in registry"):
        get_modality("unknown_modality")
