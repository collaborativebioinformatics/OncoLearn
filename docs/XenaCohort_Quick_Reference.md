# XenaCohort Quick Reference

The `XenaCohort` class provides convenient methods for loading and working with TCGA data from the UCSC Xena Browser.

## Basic Usage

```python
from oncolearn.api.xenabrowser import XenaCohortBuilder

# Build a cohort
builder = XenaCohortBuilder()
cohort = builder.build_cohort("BRCA")

# Load clinical data
clinical_df = cohort.clinical()
```

## Available Data Loading Methods

### Clinical Data
```python
clinical_df = cohort.clinical()
# Loads all clinical datasets (phenotype, survival, PAM50, etc.)
```

### Gene Expression Data
```python
mrna_df = cohort.mrna_seq()
# Loads mRNA sequencing data (FPKM, TPM, counts, etc.)
```

### Protein Expression
```python
protein_df = cohort.protein()
# Loads RPPA protein expression data
```

### DNA Methylation
```python
methylation_df = cohort.methylation()
# Loads methylation450, methylation27 data
```

### Copy Number Variation
```python
cnv_df = cohort.cnv()
# Loads CNV data (gene-level and segment-level)
```

### Somatic Mutations
```python
mutation_df = cohort.mutation()
# Loads somatic mutation data (SNPs and indels)
```

### microRNA Expression
```python
mirna_df = cohort.mirna_seq()
# Loads microRNA sequencing data
```

### Genomics Data (ATAC-seq, etc.)
```python
genomics_df = cohort.genomics()
# Loads ATAC-seq and other genomics data
```

## Merging Multiple Datasets

By default, multiple datasets within a category are concatenated vertically. You can merge them on a common column instead:

```python
# Merge clinical datasets on 'sample' column
clinical_df = cohort.clinical(merge_on='sample')

# Merge mRNA datasets on 'sample_id'
mrna_df = cohort.mrna_seq(merge_on='sample_id')
```

## Working with Multiple Cohorts

```python
builder = XenaCohortBuilder()

# Load multiple cohorts
brca = builder.build_cohort("BRCA")
luad = builder.build_cohort("LUAD")

# Compare clinical data across cohorts
brca_clinical = brca.clinical()
luad_clinical = luad.clinical()
```

## Cohort Properties

```python
cohort.name          # Full name (e.g., "TCGA-BRCA")
cohort.code          # Short code (e.g., "BRCA")
cohort.datasets      # List of all Dataset objects
cohort.base_dir      # Directory where data is stored
```

## Downloading Data

```python
# Download all datasets for a cohort
cohort.download()

# Download with options
cohort.download(
    output_dir="custom/path",
    extract=True,              # Extract .gz files
    download_mapping=True,     # Download gene mapping files
    download_raw=True          # Download raw data files
)
```

## Getting Dataset Information

```python
# List all datasets
dataset_names = cohort.list_datasets()

# Get datasets by category
from oncolearn.api.dataset import DataCategory
clinical_datasets = cohort.get_datasets_by_category(DataCategory.CLINICAL)

# Get specific dataset
dataset = cohort.get_dataset("TCGA-BRCA.clinical.tsv")
```

## Data Source Tracking

All loaded DataFrames include a `_source_dataset` column that indicates which dataset each row came from:

```python
clinical_df = cohort.clinical()
print(clinical_df['_source_dataset'].unique())
# Output: ['brca/pam50', 'TCGA-BRCA.clinical.tsv', 'TCGA-BRCA.survival.tsv']
```

## Example: Complete Analysis Workflow

```python
from oncolearn.api.xenabrowser import XenaCohortBuilder
import pandas as pd

# 1. Build cohort
builder = XenaCohortBuilder()
cohort = builder.build_cohort("BRCA")

# 2. Download data (if not already downloaded)
# cohort.download()

# 3. Load different data types
clinical = cohort.clinical()
mrna = cohort.mrna_seq()
protein = cohort.protein()

# 4. Filter and analyze
if clinical is not None:
    # Get patients with specific characteristics
    stage_3_4 = clinical[clinical['ajcc_pathologic_stage.diagnoses'].str.contains('Stage III|Stage IV', na=False)]
    print(f"Stage III/IV patients: {len(stage_3_4)}")

# 5. Integrate multiple data types
if clinical is not None and protein is not None:
    # Merge clinical and protein data
    integrated = pd.merge(clinical, protein, on='sample', how='inner')
    print(f"Integrated data shape: {integrated.shape}")
```

## Available Cohorts

To see all available cohorts:

```python
cohorts = builder.list_available_cohorts()
print(cohorts)
```

## Error Handling

```python
# Check if data exists before loading
clinical_df = cohort.clinical()
if clinical_df is None:
    print("Clinical data not available. Run cohort.download()")
else:
    print(f"Loaded {len(clinical_df)} rows")
```

## Performance Tips

1. **Selective Loading**: Only load the data types you need
2. **Merging**: Use `merge_on` parameter judiciously - it can create large DataFrames
3. **File Format**: Data is stored as TSV files which load faster than other formats
4. **Memory**: Large datasets (methylation, CNV) may require significant RAM

## Advanced: Custom Base Directory

```python
cohort = builder.build_cohort("BRCA")
cohort.base_dir = Path("/custom/data/path/TCGA-BRCA")

# Now loading methods will look in the custom directory
clinical_df = cohort.clinical()
```
