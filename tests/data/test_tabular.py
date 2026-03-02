import pytest
import pandas as pd
import torch
from pathlib import Path

from oncolearn.data.modalities.tabular.parsers.xenabrowser_parser import XenabrowserParser
from oncolearn.data.modalities.tabular.dataset import TabularDataset, TabularDataModule


# ---------------------------------------------------------------------------
# XenabrowserParser
# ---------------------------------------------------------------------------

def test_xenabrowser_parser_can_parse():
    assert XenabrowserParser.can_parse(Path("clinical_data.tsv")) is True
    assert XenabrowserParser.can_parse(Path("clinical_data.csv")) is False
    assert XenabrowserParser.can_parse(Path("clinical_data.txt")) is False


def test_xenabrowser_parser_renames_sample_to_patient_id(tmp_path):
    """'sample' column is renamed to 'patient_id'."""
    tsv = tmp_path / "gene_expr.tsv"
    pd.DataFrame({
        "sample": ["TCGA-A1-A0SE-01", "TCGA-A2-A0CX-01"],
        "gene_A": [1.2, 0.4],
        "gene_B": [0.1, 3.4],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert "patient_id" in df.columns
    assert "sample" not in df.columns


def test_xenabrowser_parser_retains_patient_id(tmp_path):
    """Files that already have 'patient_id' are left unchanged."""
    tsv = tmp_path / "clinical.tsv"
    pd.DataFrame({
        "patient_id": ["TCGA-A1-A0SE"],
        "age": [54],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert "patient_id" in df.columns
    assert len(df) == 1


def test_xenabrowser_parser_truncates_tcga_ids(tmp_path):
    """Long TCGA sample barcodes are truncated to the 12-char patient ID."""
    tsv = tmp_path / "gene_expr.tsv"
    pd.DataFrame({
        "sample": ["TCGA-A1-A0SE-01A-11R-A084-07"],
        "gene_A": [2.5],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert df["patient_id"].iloc[0] == "TCGA-A1-A0SE"


def test_xenabrowser_parser_encodes_subtype_column(tmp_path):
    """'Subtype' column is label-encoded and renamed to 'label'."""
    tsv = tmp_path / "clinical.tsv"
    pd.DataFrame({
        "sample": ["TCGA-01", "TCGA-02", "TCGA-03"],
        "Subtype": ["LumA", "Her2", "LumA"],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert "label" in df.columns
    assert "Subtype" not in df.columns
    # Same subtype must map to the same integer
    assert df.loc[df["patient_id"] == "TCGA-01", "label"].iloc[0] == \
           df.loc[df["patient_id"] == "TCGA-03", "label"].iloc[0]


# ---------------------------------------------------------------------------
# TabularDataset
# ---------------------------------------------------------------------------

def _make_gene_df(n=10, n_genes=5, prefix="TCGA-XX-000"):
    """Helper: build a minimal gene-expression DataFrame."""
    ids = [f"{prefix}{i:02d}" for i in range(n)]
    data = {f"gene_{i}": [float(i + j) for j in range(n)] for i in range(n_genes)}
    data["patient_id"] = ids
    return pd.DataFrame(data)


def test_tabular_dataset_len():
    df = _make_gene_df(n=8)
    ds = TabularDataset(df)
    assert len(ds) == 8


def test_tabular_dataset_returns_tensor():
    df = _make_gene_df(n=4, n_genes=3)
    ds = TabularDataset(df)
    item = ds[0]

    assert "tabular" in item
    assert isinstance(item["tabular"], torch.Tensor)
    assert item["tabular"].shape == (3,)
    assert item["tabular"].dtype == torch.float32


def test_tabular_dataset_fills_na_with_zero():
    df = pd.DataFrame({
        "patient_id": ["TCGA-01", "TCGA-02"],
        "gene_A": [1.0, None],
        "gene_B": [None, 2.0],
    })
    ds = TabularDataset(df)

    assert ds[0]["tabular"][1].item() == pytest.approx(0.0)
    assert ds[1]["tabular"][0].item() == pytest.approx(0.0)


def test_tabular_dataset_truncates_tcga_ids():
    df = pd.DataFrame({
        "patient_id": ["TCGA-A1-A0SE-01A-11R"],
        "gene_A": [1.0],
    })
    ds = TabularDataset(df)
    assert ds.patient_ids[0] == "TCGA-A1-A0SE"


def test_tabular_dataset_get_keys_matches_patient_ids():
    df = _make_gene_df(n=6)
    ds = TabularDataset(df)
    assert ds.get_keys() == ds.patient_ids


def test_tabular_dataset_raises_on_missing_patient_id_col():
    df = pd.DataFrame({"gene_A": [1.0, 2.0]})
    with pytest.raises(KeyError):
        TabularDataset(df)


def test_tabular_dataset_label_col_excluded_from_features():
    df = pd.DataFrame({
        "patient_id": ["TCGA-01", "TCGA-02"],
        "gene_A": [1.0, 2.0],
        "label": [0, 1],
    })
    ds = TabularDataset(df, label_col="label")

    assert "label" not in ds.feature_cols
    assert ds[0]["tabular"].shape == (1,)  # only gene_A
    assert ds[0]["label"].item() == 0


# ---------------------------------------------------------------------------
# TabularDataModule — BRCA-style merging
# ---------------------------------------------------------------------------

def _write_gene_tsv(path: Path, patient_ids, genes):
    """Write a gene-expression TSV in Xenabrowser format (sample column)."""
    data = {"sample": patient_ids}
    for i, gene in enumerate(genes):
        data[gene] = [float(i + j) for j in range(len(patient_ids))]
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def _write_clinical_tsv(path: Path, patient_ids, subtypes=None):
    """Write a clinical TSV with optional Subtype column."""
    data = {
        "sample": patient_ids,
        "age": [40 + j for j in range(len(patient_ids))],
    }
    if subtypes:
        data["Subtype"] = subtypes
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


@pytest.fixture()
def brca_data_dir(tmp_path):
    """
    Populate a temp directory that mimics what XenaCohortBuilder would download
    for the BRCA cohort: one gene-expression TSV and one clinical TSV.

    Patients TCGA-A1-0001 through TCGA-A1-0006 appear in both files;
    TCGA-A1-0007 / TCGA-A1-0008 are clinical-only (should be dropped on merge).
    """
    cohort_dir = tmp_path / "xenabrowser" / "BRCA"
    cohort_dir.mkdir(parents=True)

    shared = [f"TCGA-A1-000{i}" for i in range(1, 7)]     # 6 patients in both
    clinical_only = [f"TCGA-A1-000{i}" for i in range(7, 9)]  # 2 clinical-only

    _write_gene_tsv(
        cohort_dir / "gene_expression.tsv",
        patient_ids=shared,
        genes=["TP53", "BRCA1", "ERBB2"],
    )
    _write_clinical_tsv(
        cohort_dir / "clinical.tsv",
        patient_ids=shared + clinical_only,
        subtypes=["LumA", "LumB", "Her2", "LumA", "Basal", "LumB", "LumA", "Her2"],
    )

    return tmp_path / "xenabrowser"


def test_tabular_datamodule_setup_parses_all_tsv_files(brca_data_dir):
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    assert dm.master_df is not None
    assert not dm.master_df.empty


def test_tabular_datamodule_inner_join_drops_clinical_only_patients(brca_data_dir):
    """Only patients present in ALL modality files should survive the merge."""
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    # 6 shared patients survive; 2 clinical-only are dropped
    assert len(dm.master_df) == 6


def test_tabular_datamodule_merged_df_has_columns_from_all_files(brca_data_dir):
    """master_df should contain columns from both the gene and clinical TSVs."""
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    cols = dm.master_df.columns.tolist()
    assert "TP53" in cols
    assert "BRCA1" in cols
    assert "age" in cols


def test_tabular_datamodule_train_val_split(brca_data_dir):
    """80/20 split: train gets ≥4 samples, val gets the rest (total = 6)."""
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
        train_split=0.8,
        seed=42,
    )
    dm.setup()

    train_n = len(dm.train_dataset)
    val_n = len(dm.val_dataset)

    assert train_n + val_n == 6
    assert train_n >= 4


def test_tabular_datamodule_datasets_return_tensors(brca_data_dir):
    """Items from train_dataset should be float32 tensors."""
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert isinstance(item["tabular"], torch.Tensor)
    assert item["tabular"].dtype == torch.float32


def test_tabular_datamodule_dataloader_batches(brca_data_dir):
    """train_dataloader should yield at least one batch without error."""
    dm = TabularDataModule(
        cohort_code="BRCA",
        data_dir=str(brca_data_dir),
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert "tabular" in batch
    assert batch["tabular"].ndim == 2  # (B, num_features)
