"""
Pipeline definition for TCGA-BRCA multimodal data with PAM50 subtype labels.

Uses:
- Clinical data from cBioPortal Pan-Can Atlas 2018 (has SUBTYPE / PAM50 column).
- mRNA from cBioPortal Firehose Legacy (raw RNA-seq RSEM counts, log2-normalised).
- DICOM imaging from TCIA.

Usage in YAML config::

    data:
      pipeline: data/configs/modeling/multimodal/preprocessing/tcga_brca_cbioportal_pam50.py
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
from oncolearn.data.pipeline.transforms import make_subtype_transform

# PAM50 class mapping
_PAM50_CLASSES = {
    "BRCA_Basal": 0,
    "BRCA_Her2": 1,
    "BRCA_LumA": 2,
    "BRCA_LumB": 3,
    "BRCA_Normal": 4,
}

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
    label_col="SUBTYPE",
    label_transform=make_subtype_transform(_PAM50_CLASSES),
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
