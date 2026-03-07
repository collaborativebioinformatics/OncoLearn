"""
Tests for XenabrowserParser, TabularDataset (base), and GeneDataModule.

These replace the old test_tabular.py tests that referenced the now-removed
monolithic TabularDataModule / TabularDataset from tabular.dataset.
"""
import pytest
import pandas as pd
import torch
from pathlib import Path

from oncolearn.data.modalities.tabular.parsers.xenabrowser_parser import XenabrowserParser
from oncolearn.data.modalities.tabular.parsers.gene_parser import GeneParser
from oncolearn.data.modalities.tabular.base import TabularDataset
from oncolearn.data.modalities.tabular.gene import GeneDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gene_tsv(path: Path, patient_ids, genes):
    """Write a gene-expression TSV in XenaBrowser format (sample column)."""
    data = {"sample": patient_ids}
    for i, gene in enumerate(genes):
        data[gene] = [float(i + j) for j in range(len(patient_ids))]
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def _make_gene_df(n=10, n_genes=5, prefix="TCGA-XX-000"):
    ids = [f"{prefix}{i:02d}" for i in range(n)]
    data = {f"gene_{i}": [float(i + j) for j in range(n)] for i in range(n_genes)}
    data["patient_id"] = ids
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# XenabrowserParser
# ---------------------------------------------------------------------------

def test_xenabrowser_parser_can_parse():
    assert XenabrowserParser.can_parse(Path("clinical_data.tsv")) is True
    assert XenabrowserParser.can_parse(Path("clinical_data.csv")) is False
    assert XenabrowserParser.can_parse(Path("clinical_data.txt")) is False


def test_xenabrowser_parser_renames_sample_to_patient_id(tmp_path):
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
    tsv = tmp_path / "clinical.tsv"
    pd.DataFrame({
        "patient_id": ["TCGA-A1-A0SE"],
        "age": [54],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert "patient_id" in df.columns
    assert len(df) == 1


def test_xenabrowser_parser_truncates_tcga_ids(tmp_path):
    tsv = tmp_path / "gene_expr.tsv"
    pd.DataFrame({
        "sample": ["TCGA-A1-A0SE-01A-11R-A084-07"],
        "gene_A": [2.5],
    }).to_csv(tsv, sep="\t", index=False)

    df = XenabrowserParser.parse(tsv)

    assert df["patient_id"].iloc[0] == "TCGA-A1-A0SE"


def test_gene_parser_encodes_subtype_column(tmp_path):
    """GeneParser encodes Subtype → integer label; XenabrowserParser leaves it raw."""
    tsv = tmp_path / "pam50.tsv"
    pd.DataFrame({
        "sample": ["TCGA-01", "TCGA-02", "TCGA-03"],
        "Subtype": ["LumA", "Her2", "LumA"],
    }).to_csv(tsv, sep="\t", index=False)

    df = GeneParser.parse(tsv)

    assert "label" in df.columns
    assert "Subtype" not in df.columns
    # Same subtype → same integer label
    assert df.loc[df["patient_id"] == "TCGA-01", "label"].iloc[0] == \
           df.loc[df["patient_id"] == "TCGA-03", "label"].iloc[0]


# ---------------------------------------------------------------------------
# TabularDataset (shared base class)
# ---------------------------------------------------------------------------

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

    assert ds.feature_dim == 1  # only gene_A (label excluded from features)
    assert ds[0]["tabular"].shape == (1,)
    assert ds[0]["label"].item() == 0


# ---------------------------------------------------------------------------
# GeneDataModule — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gene_data_dir(tmp_path):
    """
    Two gene-expression TSVs in a BRCA cohort directory.

    gene_expr1.tsv: patients 1-6 (TCGA-A1-0001 to 0006)
    gene_expr2.tsv: patients 1-4 + 7-8 (overlap = 1-4; patients 5,6 are file1-only)

    After inner join on patient_id, only patients 1-4 survive.
    """
    cohort_dir = tmp_path / "xenabrowser" / "BRCA"
    cohort_dir.mkdir(parents=True)

    file1_ids = [f"TCGA-A1-000{i}" for i in range(1, 7)]  # 1-6
    file2_ids = [f"TCGA-A1-000{i}" for i in range(1, 5)] + \
                [f"TCGA-A1-000{i}" for i in range(7, 9)]   # 1-4 + 7-8

    _write_gene_tsv(cohort_dir / "gene_expr1.tsv", file1_ids, ["TP53", "BRCA1"])
    _write_gene_tsv(cohort_dir / "gene_expr2.tsv", file2_ids, ["ERBB2", "MKI67"])

    return tmp_path / "xenabrowser"


# ---------------------------------------------------------------------------
# GeneDataModule tests
# ---------------------------------------------------------------------------

def test_gene_datamodule_setup_loads_files(gene_data_dir):
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    assert hasattr(dm, "train_dataset")
    assert len(dm.train_dataset) > 0


def test_gene_datamodule_inner_join_drops_file_exclusive_patients(gene_data_dir):
    """Only patients present in ALL gene files survive the inner-join merge."""
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
    )
    df = dm._load_df()

    # 4 patients are in both files; patients 5,6 (file1-only) and 7,8 (file2-only) dropped
    assert len(df) == 4


def test_gene_datamodule_merged_df_has_columns_from_all_files(gene_data_dir):
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
    )
    df = dm._load_df()

    assert "TP53" in df.columns
    assert "BRCA1" in df.columns
    assert "ERBB2" in df.columns
    assert "MKI67" in df.columns


def test_gene_datamodule_train_val_split(gene_data_dir):
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
        train_split=0.75,
        seed=42,
    )
    dm.setup()

    assert len(dm.train_dataset) + len(dm.val_dataset) == 4


def test_gene_datamodule_datasets_return_tensors(gene_data_dir):
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
        batch_key="gene",
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert "gene" in item
    assert isinstance(item["gene"], torch.Tensor)
    assert item["gene"].dtype == torch.float32


def test_gene_datamodule_dataloader_batches(gene_data_dir):
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=2,
        num_workers=0,
        batch_key="gene",
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert "gene" in batch
    assert batch["gene"].ndim == 2  # (B, num_features)


def test_gene_datamodule_dotted_batch_key(gene_data_dir):
    """batch_key='oncolearn.modality.gene' is routed through the dataset correctly."""
    dm = GeneDataModule(
        cohort_code="BRCA",
        base_directory=str(gene_data_dir),
        files=["gene_expr1.tsv", "gene_expr2.tsv"],
        batch_size=4,
        num_workers=0,
        batch_key="oncolearn.modality.gene",
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert "oncolearn.modality.gene" in item
