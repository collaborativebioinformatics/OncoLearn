"""Tests for ImageDataModule params."""

import pytest

pytorch_lightning = pytest.importorskip("pytorch_lightning", reason="requires pytorch_lightning")


# ---------------------------------------------------------------------------
# ImageDataModule
# ---------------------------------------------------------------------------

def test_image_base_directory():
    from oncolearn.data.modules.image import ImageDataModule
    from pathlib import Path

    dm = ImageDataModule(base_directory="data/custom_tcia")
    assert dm.data_dir == Path("data/custom_tcia")


def test_image_cohort_code():
    from oncolearn.data.modules.image import ImageDataModule

    dm = ImageDataModule(cohort_code="LUAD")
    assert dm.cohort_name == "LUAD"


def test_image_batch_key_default():
    from oncolearn.data.modules.image import ImageDataModule

    dm = ImageDataModule()
    assert dm.batch_key == "image"


def test_image_batch_key_custom():
    from oncolearn.data.modules.image import ImageDataModule

    dm = ImageDataModule(batch_key="oncolearn.modality.image")
    assert dm.batch_key == "oncolearn.modality.image"
