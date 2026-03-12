"""
Pipeline definition for TCGA-BRCA multimodal data with AJCC stage labels.

Uses:
- Clinical data from cBioPortal Pan-Can Atlas 2018 (AJCC_PATHOLOGIC_TUMOR_STAGE column).
- mRNA from cBioPortal Firehose Legacy (raw RNA-seq RSEM counts, log2-normalised).
- DICOM imaging from TCIA.

Usage in YAML config::

    data:
      pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_cbioportal_stage.py
"""
from oncolearn.data.pipeline import (
    DataSource,
    Dataset,
    ImageModality,
    Log2Normalization,
    Sequence,
    Load,
    TabularModality,
)
from oncolearn.data.pipeline.transforms import map_ajcc_stage

pan_can_source = DataSource(
    config="data/configs/sources/cbioportal/brca_tcga_pan_can_atlas_2018.yaml",
    base_dir="data/sources/cbioportal",
)

brca_tcga_source = DataSource(
    config="data/configs/sources/cbioportal/brca_tcga.yaml",
    base_dir="data/sources/cbioportal",
)

clinical = TabularModality(
    name="oncolearn.modality.clinical",
    pipeline=Load("clinical_patient", source=pan_can_source),
    label_col="AJCC_PATHOLOGIC_TUMOR_STAGE",
    label_transform=map_ajcc_stage,
)

gene = TabularModality(
    name="oncolearn.modality.gene",
    pipeline=Sequence([
        Load("rna_seq_v2_mrna", source=brca_tcga_source),
        Log2Normalization(),
    ]),
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
