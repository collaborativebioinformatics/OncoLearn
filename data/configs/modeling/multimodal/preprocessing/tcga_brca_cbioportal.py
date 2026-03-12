"""
Pipeline definition for TCGA-BRCA cBioPortal data.

Defines five modalities: clinical, gene (mRNA), CNV, protein, and DICOM imaging.
Tabular modalities are joined on patient_id using the cBioPortal Firehose Legacy study.

Usage in YAML config::

    data:
      pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_cbioportal.py
"""
from oncolearn.data.pipeline import DataSource, Join, Load, TabularModality, ImageModality, Dataset, Sequence
from oncolearn.data.pipeline.transforms import map_ajcc_stage

data_source = DataSource(
    config="data/configs/sources/cbioportal/brca_tcga.yaml",
    base_dir="data/sources/cbioportal",
)

clinical = TabularModality(
    name="oncolearn.modality.clinical",
    pipeline=Load("clinical_patient", source=data_source),
    label_col="AJCC_PATHOLOGIC_TUMOR_STAGE",
    label_transform=map_ajcc_stage,
)

gene = TabularModality(
    name="oncolearn.modality.gene",
    pipeline=Sequence([
        Load("rna_seq_v2_mrna", source=data_source),
        Load("rna_seq_v2_mrna_median_Zscores", source=data_source),
        Join(on="patient_id", how="inner"),
    ]),
)

cnv = TabularModality(
    name="oncolearn.modality.cnv",
    pipeline=Sequence([
        Load("gistic", source=data_source),
        Load("linear_CNA", source=data_source),
        Join(on="patient_id", how="inner"),
    ]),
)

protein = TabularModality(
    name="oncolearn.modality.protein",
    pipeline=Sequence([
        Load("rppa", source=data_source),
        Load("rppa_Zscores", source=data_source),
        Join(on="patient_id", how="inner"),
    ]),
)

image = ImageModality(
    base_dir="data/sources/tcia",
    cohort_code="BRCA",
    n_slices=5,
)

dataset = Dataset(
    modalities=[clinical, gene, cnv, protein, image],
    name="oncolearn.datasets.multimodal",
    join_on="patient_id",
    join_strategy="inner",
)
