# UCSC Xena Browser TCGA Data Module

This module provides a clean, YAML-based configuration system for downloading TCGA cohort data from the UCSC Xena Browser.

## Structure

```
xenabrowser/
├── __init__.py              # Module exports
├── cohort_builder.py        # Builder pattern for creating cohorts from YAML
├── xena_dataset.py          # Generic dataset class
├── utils.py                 # Download utilities
└── configs/                 # YAML configuration files
    ├── acc.yaml
    ├── blca.yaml
    ├── brca.yaml
    └── ... (all TCGA cohorts)
```

## Usage

### Basic Usage

```python
from oncolearn.data.xenabrowser import CohortBuilder

# Create a builder
builder = CohortBuilder()

# Build and download a cohort
brca_cohort = builder.build_cohort("BRCA")
brca_cohort.download()  # Downloads all BRCA datasets

# Download to a specific directory
brca_cohort.download(output_dir="my_data/brca")
```

### List Available Cohorts

```python
from oncolearn.data.xenabrowser import CohortBuilder

builder = CohortBuilder()
cohorts = builder.list_available_cohorts()
print(cohorts)  # ['ACC', 'BLCA', 'BRCA', ...]
```

### Access Individual Datasets

```python
from oncolearn.data.xenabrowser import CohortBuilder

builder = CohortBuilder()
brca_cohort = builder.build_cohort("BRCA")

# List all datasets
dataset_names = brca_cohort.list_datasets()
print(dataset_names)

# Download a specific dataset
gene_expr = brca_cohort.get_dataset("BRCA Gene Expression (HiSeq)")
gene_expr.download("my_data/brca/gene_expression")
```

### Filter Datasets by Category

```python
from oncolearn.data.xenabrowser import CohortBuilder
from oncolearn.data.dataset import DataCategory

builder = CohortBuilder()
brca_cohort = builder.build_cohort("BRCA")

# Get all clinical datasets
clinical_datasets = brca_cohort.get_datasets_by_category(DataCategory.CLINICAL)

# Get all mutation datasets
mutation_datasets = brca_cohort.get_datasets_by_category(DataCategory.MUTATION)
```

## YAML Configuration Format

Each cohort is defined in a YAML file with the following structure:

```yaml
cohort:
  code: BRCA
  name: TCGA-BRCA
  description: TCGA Breast Cancer cohort with multi-modal genomics data

datasets:
  - name: BRCA Gene Expression (HiSeq)
    description: Illumina HiSeq gene expression (RNAseq) data
    category: mrna_seq
    url: https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
    filename: HiSeqV2.gz
    default_subdir: TCGA-BRCA/gene_expression
  
  # ... more datasets
```

## Data Categories

Available data categories:
- `mrna_seq`: mRNA sequencing data
- `dna_seq`: DNA sequencing data
- `mirna_seq`: microRNA sequencing data
- `cnv`: Copy number variation
- `mutation`: Somatic mutations
- `methylation`: DNA methylation
- `protein`: Protein expression
- `clinical`: Clinical/phenotype data
- `transcriptome`: Transcriptome data
- `genomics`: General genomics
- `multimodal`: Combined data types

## Adding New Datasets

To add a new dataset to an existing cohort:

1. Open the cohort's YAML file (e.g., `configs/brca.yaml`)
2. Add a new entry to the `datasets` list:

```yaml
  - name: BRCA New Dataset
    description: Description of the new dataset
    category: appropriate_category
    url: https://download.url/dataset.gz
    filename: dataset.gz
    default_subdir: TCGA-BRCA/subdirectory
```

3. Save the file - no Python code changes needed!

## Adding New Cohorts

To add a completely new cohort:

1. Create a new YAML file in `configs/` (e.g., `newcohort.yaml`)
2. Follow the YAML structure shown above
3. The cohort will automatically be available via the builder

## Benefits of This Approach

- **Declarative**: Dataset configurations are defined in YAML, not code
- **No Code Duplication**: Single generic implementation for all datasets
- **Easy to Maintain**: Adding/modifying datasets only requires YAML changes
- **Type-Safe**: Uses enum for data categories
- **Extensible**: Easy to add new cohorts or datasets
- **Clean**: Separation of data and logic
