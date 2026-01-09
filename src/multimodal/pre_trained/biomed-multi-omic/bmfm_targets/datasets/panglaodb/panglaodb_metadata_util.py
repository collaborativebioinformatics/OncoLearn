#!/usr/bin/env python3

"""
Convert PanglaoDB metadata text files to CSV files and adds column names.

The original text files are available at
https://github.com/oscar-franzen/PanglaoDB
"""

from pathlib import Path

import pandas as pd


def create_dataframe_and_save(input_path: Path, output_path: Path, column_names):
    """
    Creates a DataFrame from a text file and saves it as a CSV file.

    Args:
    ----
        input_path (Path): The path to the input text file.
        output_path (Path): The path to save the output CSV file.
        column_names (list): The column names of the DataFrame.
    """
    data = pd.read_csv(input_path, header=None, names=column_names, na_values=r"\N")
    data.to_csv(output_path, index=False)


def convert_panglao_metadata_to_csv(file_to_column_names, input_dir, output_dir=None):
    """
    Converts the PanglaoDB metadata files from text files to CSV files and
    adds column names. The original text files are available at
    https://github.com/oscar-franzen/PanglaoDB.

    Args:
    ----
        input_dir (Path): The directory containing the PanglaoDB metadata text files.
        output_dir (Path | None): The directory to save the converted CSV files. If None, the CSV files will be saved in the same directory as the input files.
    """
    for file, column_names in file_to_column_names.items():
        input_file_path = input_dir / file
        if output_dir is None:
            output_dir = input_dir
        output_file_path = output_dir / file.replace(".txt", ".csv")
        create_dataframe_and_save(input_file_path, output_file_path, column_names)


if __name__ == "__main__":
    file_to_column_names = {
        "metadata.txt": [
            "SRA Accession",
            "SRS Accession",
            "Tissue Origin",
            "scRNA-seq Protocol",
            "Species",
            "Sequencing Instrument",
            "Number of Expressed Genes",
            "Median Expressed Genes per Cell",
            "Number of Cell Clusters",
            "Is Tumor Sample",
            "Is Primary Adult Tissue Sample",
            "Is Cell Line Sample",
        ],
        "cell_type_abbrev.txt": ["Cell Type", "Abbreviation"],
        "cell_type_annotations.txt": [
            "SRA Accession",
            "SRS Accession",
            "Cluster Index",
            "Cell Type Annotation",
            "P-value (Hypergeometric Test)",
            "Adjusted p-value (BH)",
            "Cell Type Activity Score",
        ],
        "cell_type_annotations_markers.txt": [
            "SRA Accession",
            "SRS Accession",
            "Cluster Index",
            "Gene Symbol",
        ],
        "cell_type_desc.txt": ["Cell Type", "Description", "Synonyms"],
        "genes.txt": ["Ensembl Gene ID", "Gene Symbol"],
        "germ_layers.txt": ["Cell Type", "Germ Layer", "Organ"],
    }

    input_dir = Path(
        "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/panglao/metadata"
    )
    convert_panglao_metadata_to_csv(file_to_column_names, input_dir=input_dir)
