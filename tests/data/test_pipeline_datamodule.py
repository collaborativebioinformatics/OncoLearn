"""
Unit tests for PipelineDataModule using fixture TSV data (no downloads required).

PipelineDataModule tests require pytorch_lightning and are skipped in host env.
Transforms, load_pipeline_file, and executor tests have no such dependency.
"""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from oncolearn.data.pipeline.nodes import DataSource, Dataset, Load, Modality, Sequence, Join
from oncolearn.data.pipeline.loader import load_pipeline_file
from oncolearn.data.pipeline.transforms import map_ajcc_stage
from oncolearn.data.pipeline.readers.base import BaseReader
from oncolearn.data.pipeline.executor import run

try:
    import pytorch_lightning  # noqa: F401
    _HAS_PL = True
except ImportError:
    _HAS_PL = False

requires_pl = pytest.mark.skipif(not _HAS_PL, reason="requires pytorch_lightning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clinical_tsv(tmp_path: Path) -> Path:
    """Minimal clinical TSV fixture."""
    content = textwrap.dedent("""\
        patient_id\tAJCC_PATHOLOGIC_TUMOR_STAGE\tAGE
        TCGA-A1-A0SB\tStage IIA\t55
        TCGA-A1-A0SF\tStage IIIA\t60
        TCGA-A1-A0SH\tStage I\t45
        TCGA-A2-A0YF\tStage IV\t70
        TCGA-A2-A0YG\tStage X\t50
    """)
    p = tmp_path / "clinical.tsv"
    p.write_text(content)
    return p


@pytest.fixture
def gene_tsv(tmp_path: Path) -> Path:
    """Minimal gene TSV fixture (xenabrowser genomic matrix style)."""
    content = textwrap.dedent("""\
        sample\tTCGA-A1-A0SB\tTCGA-A1-A0SF\tTCGA-A1-A0SH\tTCGA-A2-A0YF
        MIR1\t1.0\t2.0\t3.0\t4.0
        MIR2\t0.5\t1.5\t2.5\t3.5
    """)
    p = tmp_path / "mirna.tsv"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_map_ajcc_stage_stage_i():
    assert map_ajcc_stage("Stage I") == 0
    assert map_ajcc_stage("Stage IA") == 0
    assert map_ajcc_stage("Stage IB") == 0


def test_map_ajcc_stage_stage_ii():
    assert map_ajcc_stage("Stage IIA") == 1
    assert map_ajcc_stage("Stage IIB") == 1


def test_map_ajcc_stage_stage_iii():
    assert map_ajcc_stage("Stage IIIA") == 2
    assert map_ajcc_stage("Stage III") == 2


def test_map_ajcc_stage_stage_iv():
    assert map_ajcc_stage("Stage IV") == 3


def test_map_ajcc_stage_unknown_returns_none():
    assert map_ajcc_stage("Stage X") is None
    assert map_ajcc_stage("Unknown") is None
    assert map_ajcc_stage(None) is None
    assert map_ajcc_stage(float("nan")) is None


# ---------------------------------------------------------------------------
# PipelineDataModule with a custom reader that reads fixture files
# ---------------------------------------------------------------------------

class FileReader(BaseReader):
    """Simple reader that loads named TSV fixture files directly."""
    def __init__(self, files: dict):
        self._files = files  # name → Path

    def read(self, name: str) -> pd.DataFrame:
        path = self._files[name]
        df = pd.read_csv(str(path), sep="\t")
        # Transpose genomic matrix if needed
        sample_cols = [c for c in df.columns[1:5] if str(c).startswith("TCGA-")]
        if len(sample_cols) >= 2:
            id_col = df.columns[0]
            df = df.set_index(id_col).T.reset_index()
            df = df.rename(columns={"index": "patient_id"})
        return df


def _make_clinical_modality(clinical_tsv: Path) -> Modality:
    ds = DataSource(config="fixture", base_dir=str(clinical_tsv.parent), reader="xenabrowser")
    return Modality(
        name="oncolearn.modality.clinical",
        pipeline=Load(clinical_tsv.name, source=ds),
        label_col="AJCC_PATHOLOGIC_TUMOR_STAGE",
        label_transform=map_ajcc_stage,
    )


def _make_gene_modality(gene_tsv: Path) -> Modality:
    ds = DataSource(config="fixture", base_dir=str(gene_tsv.parent), reader="xenabrowser")
    return Modality(
        name="oncolearn.modality.gene",
        pipeline=Load(gene_tsv.name, source=ds),
    )


@requires_pl
def test_pipeline_datamodule_setup_clinical(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    # Stage X patient is dropped; 4 valid patients remain
    assert dm._full_dataset is not None
    assert len(dm._full_dataset) == 4
    assert dm._full_dataset.labels is not None


@requires_pl
def test_pipeline_datamodule_labels_are_ints(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()
    labels = dm._full_dataset.labels.tolist()
    assert all(isinstance(l, int) for l in labels)
    assert set(labels) == {0, 1, 2, 3}  # I, II, III, IV


@requires_pl
def test_pipeline_datamodule_setup_full(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
    )
    dm.setup_full()
    assert dm.full_dataset is not None
    assert len(dm.full_dataset) == 4


@requires_pl
def test_pipeline_datamodule_batch_key(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()
    sample = dm._full_dataset[0]
    assert "oncolearn.modality.clinical" in sample


@requires_pl
def test_pipeline_datamodule_name_attribute(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
    )
    assert dm.name == "oncolearn.modality.clinical"


@requires_pl
def test_pipeline_datamodule_train_val_test_split(clinical_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_clinical_modality(clinical_tsv),
        batch_size=2,
        num_workers=0,
        train_split=0.5,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert total == 4


@requires_pl
def test_pipeline_datamodule_gene_no_labels(gene_tsv):
    from oncolearn.data.modules.base import PipelineDataModule
    dm = PipelineDataModule.from_modality(
        modality=_make_gene_modality(gene_tsv),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()
    assert dm._full_dataset.labels is None
    assert dm._full_dataset.feature_dim == 2  # MIR1, MIR2


# ---------------------------------------------------------------------------
# load_pipeline_file
# ---------------------------------------------------------------------------

def test_load_pipeline_file(tmp_path):
    pipeline_content = textwrap.dedent("""\
        from oncolearn.data.pipeline import DataSource, Load, Modality, Dataset
        ds = DataSource(config="xenabrowser", base_dir="data/sources/xenabrowser/TCGA-BRCA")
        gene = Modality(
            name="oncolearn.modality.gene",
            pipeline=Load("TCGA-BRCA.mirna.tsv", source=ds),
        )
        dataset = Dataset(modalities=[gene])
    """)
    p = tmp_path / "test_pipeline.py"
    p.write_text(pipeline_content)

    dataset = load_pipeline_file(str(p))
    assert len(dataset.modalities) == 1
    assert dataset.modalities[0].name == "oncolearn.modality.gene"


def test_load_pipeline_file_missing_dataset_attr(tmp_path):
    p = tmp_path / "bad_pipeline.py"
    p.write_text("x = 1  # no 'dataset' variable\n")

    with pytest.raises(AttributeError, match="dataset"):
        load_pipeline_file(str(p))


def test_load_pipeline_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pipeline_file("/no/such/pipeline.py")
