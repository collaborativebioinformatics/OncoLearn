"""
Test script to demonstrate XenaCohort data loading functionality.
"""

from oncolearn.api.xenabrowser import XenaCohortBuilder

# Build BRCA cohort
builder = XenaCohortBuilder()
cohort = builder.build_cohort("BRCA")

print(f"Loaded cohort: {cohort}")
print(f"Total datasets: {len(cohort.datasets)}")
print()

# List datasets by category
print("Datasets by category:")
from oncolearn.api.dataset import DataCategory

for category in DataCategory:
    datasets = cohort.get_datasets_by_category(category)
    if datasets:
        print(f"\n{category.value.upper()} ({len(datasets)} datasets):")
        for ds in datasets[:3]:  # Show first 3
            print(f"  - {ds.name}")
        if len(datasets) > 3:
            print(f"  ... and {len(datasets) - 3} more")

print("\n" + "="*60)
print("Loading clinical data...")
print("="*60)

# Load clinical data
clinical_df = cohort.clinical()

if clinical_df is not None:
    print(f"\nClinical data shape: {clinical_df.shape}")
    print(f"Columns: {len(clinical_df.columns)}")
    print(f"\nFirst few columns:")
    print(clinical_df.columns[:10].tolist())
    print(f"\nFirst few rows:")
    print(clinical_df.head())
else:
    print("No clinical data available or files not downloaded yet.")
    print("\nTo download the data, run:")
    print("  cohort.download()")

print("\n" + "="*60)
print("Other available loading methods:")
print("="*60)
print("""
- cohort.mrna_seq()      # Load mRNA sequencing data
- cohort.protein()       # Load protein expression data
- cohort.methylation()   # Load DNA methylation data
- cohort.cnv()           # Load copy number variation data
- cohort.mutation()      # Load somatic mutation data
- cohort.mirna_seq()     # Load microRNA sequencing data
- cohort.genomics()      # Load general genomics data (e.g., ATAC-seq)

All methods support optional merge_on parameter to merge multiple datasets:
  cohort.clinical(merge_on='sample')
""")
