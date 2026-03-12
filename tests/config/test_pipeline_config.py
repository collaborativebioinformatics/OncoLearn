"""
Tests for pipeline config round-trip and reader selection.
"""
import textwrap
from pathlib import Path

import pytest
import yaml

from oncolearn.config.loader import load_config
from oncolearn.data.pipeline.loader import load_pipeline_file
from oncolearn.data.pipeline.nodes import DataSource, Dataset


# ---------------------------------------------------------------------------
# Config round-trip: pipeline path survives save/load
# ---------------------------------------------------------------------------

def test_pipeline_path_survives_roundtrip(tmp_path):
    from oncolearn.config.loader import save_config
    from oncolearn.config.schema import DataConfig, ModelConfig, OncoLearnConfig

    original = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        data=DataConfig(pipeline="data/configs/modeling/multimodal/preprocessing/tcga_brca_cbioportal.py"),
    )
    out = tmp_path / "cfg.yaml"
    save_config(original, out)
    restored = load_config(out)
    assert restored.data.pipeline == original.data.pipeline


def test_splits_dir_null_survives_roundtrip(tmp_path):
    from oncolearn.config.loader import save_config
    from oncolearn.config.schema import DataConfig, ModelConfig, OncoLearnConfig

    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        data=DataConfig(pipeline="p.py", splits_dir=None),
    )
    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    restored = load_config(out)
    assert restored.data.splits_dir is None


def test_splits_dir_set_survives_roundtrip(tmp_path):
    from oncolearn.config.loader import save_config
    from oncolearn.config.schema import DataConfig, ModelConfig, OncoLearnConfig

    cfg = OncoLearnConfig(
        model=ModelConfig(name="gated_late_fusion"),
        data=DataConfig(pipeline="p.py", splits_dir="data/splits/fold_0"),
    )
    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    restored = load_config(out)
    assert restored.data.splits_dir == "data/splits/fold_0"


# ---------------------------------------------------------------------------
# Reader auto-detection
# ---------------------------------------------------------------------------

def test_auto_reader_detects_cbioportal():
    from oncolearn.data.pipeline.loader import _make_reader
    from oncolearn.data.pipeline.nodes import Load, Modality
    from oncolearn.data.pipeline.readers.cbioportal import CbioPortalReader

    ds = DataSource(config="data/configs/sources/cbioportal/brca_tcga.yaml", base_dir="data/sources/cbioportal")
    modality = Modality(name="oncolearn.modality.clinical", pipeline=Load("clinical_patient", source=ds))
    reader = _make_reader(modality)
    assert isinstance(reader, CbioPortalReader)


def test_auto_reader_detects_xenabrowser():
    from oncolearn.data.pipeline.loader import _make_reader
    from oncolearn.data.pipeline.nodes import Load, Modality
    from oncolearn.data.pipeline.readers.xenabrowser import XenabrowserReader

    ds = DataSource(config="xenabrowser", base_dir="data/sources/xenabrowser/TCGA-BRCA")
    modality = Modality(name="oncolearn.modality.gene", pipeline=Load("TCGA-BRCA.mirna.tsv", source=ds))
    reader = _make_reader(modality)
    assert isinstance(reader, XenabrowserReader)


def test_explicit_reader_cbioportal():
    from oncolearn.data.pipeline.loader import _make_reader
    from oncolearn.data.pipeline.nodes import Load, Modality
    from oncolearn.data.pipeline.readers.cbioportal import CbioPortalReader

    ds = DataSource(config="some_other_path.yaml", base_dir="data", reader="cbioportal")
    modality = Modality(name="mod", pipeline=Load("x", source=ds))
    reader = _make_reader(modality)
    assert isinstance(reader, CbioPortalReader)


# ---------------------------------------------------------------------------
# load_pipeline_file — cBioPortal pipeline
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
PIPELINE_DIR = REPO_ROOT / "data" / "configs" / "modeling" / "multimodal" / "preprocessing"


def test_load_cbioportal_pipeline_file():
    dataset = load_pipeline_file(str(PIPELINE_DIR / "tcga_brca_cbioportal.py"))
    assert isinstance(dataset, Dataset)
    modality_names = [m.name for m in dataset.modalities]
    assert "oncolearn.modality.clinical" in modality_names
    assert "oncolearn.modality.gene" in modality_names
    assert "oncolearn.modality.cnv" in modality_names
    assert "oncolearn.modality.protein" in modality_names


def test_load_xenabrowser_pipeline_file():
    dataset = load_pipeline_file(str(PIPELINE_DIR / "tcga_brca_xenabrowser.py"))
    assert isinstance(dataset, Dataset)
    modality_names = [m.name for m in dataset.modalities]
    assert "oncolearn.modality.clinical" in modality_names
    assert "oncolearn.modality.gene" in modality_names


def test_cbioportal_pipeline_clinical_has_label_transform():
    dataset = load_pipeline_file(str(PIPELINE_DIR / "tcga_brca_cbioportal.py"))
    clinical = next(m for m in dataset.modalities if "clinical" in m.name)
    assert clinical.label_col == "AJCC_PATHOLOGIC_TUMOR_STAGE"
    assert clinical.label_transform is not None
    # Verify the transform works
    assert clinical.label_transform("Stage IIA") == 1
