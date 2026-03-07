"""
Tests for ClinicalDataModule and ClinicalParser.
"""
import pytest
import pandas as pd
import torch
from pathlib import Path

from oncolearn.data.modalities.tabular.parsers.clinical_parser import ClinicalParser
from oncolearn.data.modalities.tabular.clinical import ClinicalDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAGE_COL = ClinicalParser.STAGE_COL


def _write_clinical_tsv(path: Path, patient_ids, stages, extra_numeric=None):
    """Write a clinical TSV with AJCC stage column and optional numeric features."""
    data = {
        "sample": patient_ids,
        STAGE_COL: stages,
    }
    if extra_numeric:
        data.update(extra_numeric)
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


@pytest.fixture()
def clinical_data_dir(tmp_path):
    """
    A BRCA cohort directory with a clinical TSV containing 8 patients.
    Two have 'Unknown' / NaN stage and should be dropped by ClinicalParser.
    """
    cohort_dir = tmp_path / "xenabrowser" / "BRCA"
    cohort_dir.mkdir(parents=True)

    patient_ids = [f"TCGA-A1-000{i}" for i in range(1, 9)]
    stages = [
        "Stage I", "Stage II", "Stage III", "Stage IV",
        "Stage I", "Stage II", "Unknown", "Not Reported",
    ]
    ages = [40, 45, 50, 55, 60, 65, 70, 75]

    _write_clinical_tsv(
        cohort_dir / "TCGA-BRCA.clinical.tsv",
        patient_ids=patient_ids,
        stages=stages,
        extra_numeric={"age_at_diagnosis": ages},
    )

    return tmp_path / "xenabrowser"


# ---------------------------------------------------------------------------
# ClinicalParser
# ---------------------------------------------------------------------------

def test_clinical_parser_can_parse():
    assert ClinicalParser.can_parse(Path("TCGA-BRCA.clinical.tsv")) is True
    assert ClinicalParser.can_parse(Path("gene_expr.tsv")) is False


def test_clinical_parser_maps_stages_to_ints(tmp_path):
    tsv = tmp_path / "TCGA-BRCA.clinical.tsv"
    _write_clinical_tsv(
        tsv,
        patient_ids=["TCGA-01", "TCGA-02", "TCGA-03", "TCGA-04"],
        stages=["Stage I", "Stage II", "Stage III", "Stage IV"],
        extra_numeric={"age": [40, 50, 60, 70]},
    )
    df = ClinicalParser.parse(tsv)

    assert "label" in df.columns
    assert set(df["label"].unique()) == {0, 1, 2, 3}


def test_clinical_parser_drops_unknown_stage(tmp_path):
    tsv = tmp_path / "TCGA-BRCA.clinical.tsv"
    _write_clinical_tsv(
        tsv,
        patient_ids=["TCGA-01", "TCGA-02", "TCGA-03"],
        stages=["Stage I", "Unknown", "Stage II"],
        extra_numeric={"age": [40, 50, 60]},
    )
    df = ClinicalParser.parse(tsv)

    # Patient TCGA-02 (Unknown stage) should be dropped
    assert len(df) == 2
    assert "TCGA-02" not in df["patient_id"].values


def test_clinical_parser_keeps_only_numeric_features(tmp_path):
    tsv = tmp_path / "TCGA-BRCA.clinical.tsv"
    pd.DataFrame({
        "sample": ["TCGA-01", "TCGA-02"],
        STAGE_COL: ["Stage I", "Stage II"],
        "age": [40, 50],
        "text_notes": ["some text", "more text"],   # non-numeric: dropped
    }).to_csv(tsv, sep="\t", index=False)

    df = ClinicalParser.parse(tsv)

    assert "age" in df.columns
    assert "text_notes" not in df.columns


# ---------------------------------------------------------------------------
# ClinicalDataModule
# ---------------------------------------------------------------------------

def test_clinical_datamodule_setup_loads_data(clinical_data_dir):
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    assert hasattr(dm, "train_dataset")
    assert len(dm.train_dataset) > 0


def test_clinical_datamodule_drops_unknown_stage_patients(clinical_data_dir):
    """Patients with Unknown/NaN stage are dropped by ClinicalParser."""
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    # 8 total, 2 with unknown stage dropped → 6 remain
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 6


def test_clinical_datamodule_train_val_split(clinical_data_dir):
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
        train_split=0.8,
        seed=42,
    )
    dm.setup()

    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 6
    assert len(dm.train_dataset) >= 4


def test_clinical_datamodule_datasets_return_tensors(clinical_data_dir):
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert "clinical" in item
    assert isinstance(item["clinical"], torch.Tensor)
    assert item["clinical"].dtype == torch.float32


def test_clinical_datamodule_items_have_labels(clinical_data_dir):
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert "label" in item
    assert isinstance(item["label"], torch.Tensor)


def test_clinical_datamodule_files_param(clinical_data_dir):
    """files=['TCGA-BRCA.clinical.tsv'] works as alias for clinical_file."""
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        files=["TCGA-BRCA.clinical.tsv"],
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    assert len(dm.train_dataset) + len(dm.val_dataset) == 6


def test_clinical_datamodule_dotted_batch_key(clinical_data_dir):
    """batch_key='oncolearn.modality.clinical' is routed through the dataset."""
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=4,
        num_workers=0,
        batch_key="oncolearn.modality.clinical",
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert "oncolearn.modality.clinical" in item


def test_clinical_datamodule_dataloader_batches(clinical_data_dir):
    dm = ClinicalDataModule(
        cohort_code="BRCA",
        base_directory=str(clinical_data_dir),
        clinical_file="TCGA-BRCA.clinical.tsv",
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert "clinical" in batch
    assert batch["clinical"].ndim == 2  # (B, num_features)
