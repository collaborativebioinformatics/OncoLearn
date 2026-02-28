import pytest
from pathlib import Path
import pandas as pd
from oncolearn.data.modalities.tabular.parsers.xenabrowser_parser import XenabrowserParser


def test_xenabrowser_parser_can_parse():
    assert XenabrowserParser.can_parse(Path("clinical_data.tsv")) is True
    assert XenabrowserParser.can_parse(Path("clinical_data.csv")) is False
    assert XenabrowserParser.can_parse(Path("clinical_data.txt")) is False


def test_xenabrowser_parser_renames_sample_to_patient_id(tmp_path):
    """Test that it correctly maps 'sample' to 'patient_id'."""
    # Create fake TSV
    tsv_file = tmp_path / "fake_xena.tsv"
    
    # Write a simple dataframe
    df = pd.DataFrame({
        "sample": ["TCGA-01", "TCGA-02"],
        "gene_A": [1.2, 0.4],
        "gene_B": [0.1, 3.4]
    })
    df.to_csv(tsv_file, sep='\t', index=False)
    
    # Parse
    parsed_df = XenabrowserParser.parse(tsv_file)
    
    assert "patient_id" in parsed_df.columns
    assert "sample" not in parsed_df.columns
    assert parsed_df["patient_id"].iloc[0] == "TCGA-01"


def test_xenabrowser_parser_retains_patient_id(tmp_path):
    """Test that it respects 'patient_id' if already present."""
    tsv_file = tmp_path / "fake_xena2.tsv"
    
    df = pd.DataFrame({
        "patient_id": ["TCGA-01"],
        "gene_A": [1.2]
    })
    df.to_csv(tsv_file, sep='\t', index=False)
    
    parsed_df = XenabrowserParser.parse(tsv_file)
    assert "patient_id" in parsed_df.columns
    assert len(parsed_df) == 1
