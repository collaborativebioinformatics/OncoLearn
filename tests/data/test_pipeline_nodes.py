"""Unit tests for pipeline DSL node dataclasses."""
import pytest

from oncolearn.data.pipeline.nodes import (
    BaseModality,
    DataSource,
    Dataset,
    ImageModality,
    Join,
    Load,
    Modality,
    Sequence,
    TabularModality,
)


def test_data_source_defaults():
    ds = DataSource(config="data/configs/sources/cbioportal/brca_tcga.yaml", base_dir="data/sources/cbioportal")
    assert ds.reader == "auto"


def test_load_stores_source():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    node = Load(name="clinical_patient", source=ds)
    assert node.name == "clinical_patient"
    assert node.source is ds


def test_join_defaults():
    j = Join()
    assert j.on == "patient_id"
    assert j.how == "inner"


def test_sequence_stores_steps():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    steps = [
        Load("rna_seq", source=ds),
        Load("rna_zscores", source=ds),
        Join(),
    ]
    seq = Sequence(steps=steps)
    assert len(seq.steps) == 3


# ---------------------------------------------------------------------------
# TabularModality (and backwards-compatible Modality alias)
# ---------------------------------------------------------------------------

def test_tabular_modality_defaults():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    m = TabularModality(
        name="oncolearn.modality.clinical",
        pipeline=Load("clinical_patient", source=ds),
    )
    assert m.label_col is None
    assert m.label_transform is None
    assert m.patient_id_col == "patient_id"
    assert isinstance(m, BaseModality)


def test_modality_alias_is_tabular_modality():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    m = Modality(
        name="oncolearn.modality.clinical",
        pipeline=Load("clinical_patient", source=ds),
    )
    assert isinstance(m, TabularModality)
    assert isinstance(m, BaseModality)


def test_modality_with_transform():
    from oncolearn.data.pipeline.transforms import map_ajcc_stage
    ds = DataSource(config="cfg.yaml", base_dir="data")
    m = TabularModality(
        name="oncolearn.modality.clinical",
        pipeline=Load("clinical_patient", source=ds),
        label_col="AJCC_PATHOLOGIC_TUMOR_STAGE",
        label_transform=map_ajcc_stage,
    )
    assert m.label_transform is map_ajcc_stage
    assert m.label_col == "AJCC_PATHOLOGIC_TUMOR_STAGE"


# ---------------------------------------------------------------------------
# ImageModality
# ---------------------------------------------------------------------------

def test_image_modality_defaults():
    im = ImageModality()
    assert im.name == "oncolearn.modality.image"
    assert im.base_dir == "data/tcia"
    assert im.cohort_code == "BRCA"
    assert isinstance(im, BaseModality)


def test_image_modality_custom():
    im = ImageModality(name="oncolearn.modality.image", base_dir="data/tcia", cohort_code="LUAD", n_slices=10)
    assert im.cohort_code == "LUAD"
    assert im.n_slices == 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_defaults():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    dataset = Dataset(
        modalities=[
            TabularModality(name="oncolearn.modality.gene", pipeline=Load("gene.tsv", source=ds))
        ]
    )
    assert dataset.join_on == "patient_id"
    assert dataset.join_strategy == "inner"
    assert dataset.name == ""
    assert len(dataset.modalities) == 1


def test_dataset_with_name():
    ds = DataSource(config="cfg.yaml", base_dir="data")
    dataset = Dataset(
        modalities=[
            TabularModality(name="oncolearn.modality.gene", pipeline=Load("gene.tsv", source=ds))
        ],
        name="oncolearn.datasets.multimodal",
    )
    assert dataset.name == "oncolearn.datasets.multimodal"
