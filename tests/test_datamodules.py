"""Tests for DataModule files/base_directory/batch_key params (Unit 4)."""

import pytest


# ---------------------------------------------------------------------------
# GeneDataModule
# ---------------------------------------------------------------------------

def test_gene_files_param():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule

    dm = GeneDataModule(files=["a.tsv", "b.tsv"])
    assert dm.features_files == ["a.tsv", "b.tsv"]


def test_gene_features_files_backward_compat():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule

    dm = GeneDataModule(features_files=["a.tsv"])
    assert dm.features_files == ["a.tsv"]


def test_gene_files_overrides_features_files():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule

    dm = GeneDataModule(files=["x.tsv"], features_files=["y.tsv"])
    assert dm.features_files == ["x.tsv"]


def test_gene_base_directory():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule
    from pathlib import Path

    dm = GeneDataModule(base_directory="data/custom", cohort_code="TEST")
    assert dm.data_dir == Path("data/custom")


def test_gene_data_dir_backward_compat():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule
    from pathlib import Path

    dm = GeneDataModule(data_dir="data/old", cohort_code="TEST")
    assert dm.data_dir == Path("data/old")


def test_gene_base_directory_overrides_data_dir():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule
    from pathlib import Path

    dm = GeneDataModule(base_directory="data/new", data_dir="data/old", cohort_code="TEST")
    assert dm.data_dir == Path("data/new")


def test_gene_batch_key_default():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule

    dm = GeneDataModule(cohort_code="TEST", files=["a.tsv"])
    assert dm.batch_key == "gene"


def test_gene_batch_key_custom():
    from oncolearn.data.modalities.tabular.gene import GeneDataModule

    dm = GeneDataModule(
        batch_key="oncolearn.modality.gene",
        cohort_code="TEST",
        files=["a.tsv"],
    )
    assert dm.batch_key == "oncolearn.modality.gene"


# ---------------------------------------------------------------------------
# ClinicalDataModule
# ---------------------------------------------------------------------------

def test_clinical_files_param():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule

    dm = ClinicalDataModule(files=["x.tsv"])
    assert dm.clinical_file == "x.tsv"


def test_clinical_files_overrides_clinical_file():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule

    dm = ClinicalDataModule(files=["override.tsv"], clinical_file="default.tsv")
    assert dm.clinical_file == "override.tsv"


def test_clinical_clinical_file_backward_compat():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule

    dm = ClinicalDataModule(clinical_file="my.tsv")
    assert dm.clinical_file == "my.tsv"


def test_clinical_base_directory():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule
    from pathlib import Path

    dm = ClinicalDataModule(base_directory="data/clin")
    assert dm.data_dir == Path("data/clin")


def test_clinical_batch_key_default():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule

    dm = ClinicalDataModule()
    assert dm.batch_key == "clinical"


def test_clinical_batch_key_custom():
    from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule

    dm = ClinicalDataModule(batch_key="oncolearn.modality.clinical")
    assert dm.batch_key == "oncolearn.modality.clinical"


# ---------------------------------------------------------------------------
# ImageDataModule
# ---------------------------------------------------------------------------

def test_image_base_directory():
    from oncolearn.data.modalities.image.dataset import ImageDataModule
    from pathlib import Path

    dm = ImageDataModule(base_directory="data/custom_tcia")
    assert dm.data_dir == Path("data/custom_tcia")


def test_image_data_dir_backward_compat():
    from oncolearn.data.modalities.image.dataset import ImageDataModule
    from pathlib import Path

    dm = ImageDataModule(data_dir="data/tcia_old")
    assert dm.data_dir == Path("data/tcia_old")


def test_image_base_directory_overrides_data_dir():
    from oncolearn.data.modalities.image.dataset import ImageDataModule
    from pathlib import Path

    dm = ImageDataModule(base_directory="data/new_tcia", data_dir="data/old_tcia")
    assert dm.data_dir == Path("data/new_tcia")


def test_image_cohort_code_alias():
    from oncolearn.data.modalities.image.dataset import ImageDataModule

    dm = ImageDataModule(cohort_code="LUAD")
    assert dm.tcia_cohort_name == "LUAD"


def test_image_batch_key_default():
    from oncolearn.data.modalities.image.dataset import ImageDataModule

    dm = ImageDataModule()
    assert dm.batch_key == "image"


def test_image_batch_key_custom():
    from oncolearn.data.modalities.image.dataset import ImageDataModule

    dm = ImageDataModule(batch_key="oncolearn.modality.image")
    assert dm.batch_key == "oncolearn.modality.image"
