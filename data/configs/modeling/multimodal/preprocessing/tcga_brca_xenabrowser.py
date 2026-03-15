"""
Pipeline definition for TCGA-BRCA XenaBrowser data.

Defines three modalities: clinical (AJCC stage labels), gene (miRNA), and DICOM imaging.

Usage in YAML config::

    data:
      pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_xenabrowser.py
"""
from oncolearn.data.pipeline import DataSource, Load, TabularModality, ImageModality, Dataset
from oncolearn.data.pipeline.transforms import map_ajcc_stage

data_source = DataSource(
    config="xenabrowser",
    base_dir="data/sources/xenabrowser/TCGA-BRCA",
)

clinical = TabularModality(
    name="oncolearn.modality.clinical",
    pipeline=Load("TCGA-BRCA.clinical.tsv", source=data_source),
    label_col="ajcc_pathologic_stage.diagnoses",
    label_transform=map_ajcc_stage,
)

gene = TabularModality(
    name="oncolearn.modality.gene",
    pipeline=Load("TCGA-BRCA.mirna.tsv", source=data_source),
)

image = ImageModality(
    base_dir="data/sources/tcia",
    cohort_code="BRCA",
    n_slices=5,
)

dataset = Dataset(
    modalities=[clinical, gene, image],
    name="oncolearn.datasets.multimodal",
    join_on="patient_id",
    join_strategy="inner",
)
