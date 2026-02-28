import pytest
import sys
from pathlib import Path
from oncolearn.data.modalities.image.loaders.dicom_loader import DicomLoader
from oncolearn.data.modalities.image.loaders.pillow_loader import PillowLoader


def test_dicom_loader_can_load():
    assert DicomLoader.can_load(Path("test_image.dcm")) is True
    assert DicomLoader.can_load(Path("test_image.dicom")) is True
    assert DicomLoader.can_load(Path("test_image.DICOM")) is True
    assert DicomLoader.can_load(Path("test_image.png")) is False


def test_pillow_loader_can_load():
    assert PillowLoader.can_load(Path("test_image.png")) is True
    assert PillowLoader.can_load(Path("test_image.jpg")) is True
    assert PillowLoader.can_load(Path("test_image.jpeg")) is True
    assert PillowLoader.can_load(Path("test_image.tiff")) is True
    assert PillowLoader.can_load(Path("test_image.dcm")) is False


def test_dicom_loader_missing_pydicom(monkeypatch):
    """Test that DicomLoader correctly raises an ImportError with a helpful message if pydicom is missing."""
    import builtins
    real_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == "pydicom":
            raise ImportError("No module named 'pydicom'")
        return real_import(name, *args, **kwargs)
        
    monkeypatch.setattr(builtins, "__import__", mock_import)
    
    with pytest.raises(ImportError, match="pydicom and SimpleITK required"):
        DicomLoader.load(Path("fake.dcm"))


def test_pillow_loader_missing_pillow(monkeypatch):
    """Test that PillowLoader correctly raises an ImportError with a helpful message if PIL is missing."""
    # We must mock PIL.Image, since PillowLoader uses `from PIL import Image`
    # However, since it's already imported at module level, monkeypatching the function inside
    # is easier.
    def mock_open(*args, **kwargs):
        raise ImportError("No module named 'PIL'")
        
    # We mock Image.open inside the loader
    import PIL.Image
    monkeypatch.setattr(PIL.Image, "open", mock_open)
    
    with pytest.raises(ImportError, match="Pillow is required for standard image files"):
        PillowLoader.load(Path("fake.png"))
