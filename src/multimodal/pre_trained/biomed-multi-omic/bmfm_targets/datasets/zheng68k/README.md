# Zheng68K PBMC Dataset

Zheng68K is a PBMC scRNA-seq dataset widely used for cell type annotation performance acessment. It contains about 68,450 cells within eleven subtypes of cells:
- CD8+ cytotoxic T cells (30.3%)
- CD8+/CD45RA+ naive cytotoxic cells (24.3%)
- CD56+ NK cells (12.8%)
- CD4+/CD25 T Reg cells (9.0%)
- CD19+ B cells (8.6%)
- CD4+/CD45RO+ memory cells (4.5%)
- CD14+ monocyte cells (4.2%)
- dendritic cells (3.1%)
- CD4+/CD45RA+/CD25- naive T cells (2.7%)
- CD34+ cells (0.4%)
- CD4+ T Helper2 cells (0.1%)

Paper: [Zheng, G. X. Y. et al. Massively parallel digital transcriptional profiling of single cells. Nat. Commun. 8, 1–12 (2017).](https://doi.org/10.1038%2Fncomms14049)

The original raw genomic data of the experiment can be downloaded from [SRP073767](https://www.ncbi.nlm.nih.gov/sra/?term=SRP073767) study. All the experiments of this study in BAM format can be accessed by this [link](https://trace.ncbi.nlm.nih.gov/Traces/?view=study&acc=SRP073767)

Protocol of gene expression generation: [10X CHROMIUM](https://www.10xgenomics.com/products/single-cell-gene-expression)

The results of cell expression can be found in [Fresh 68k PBMCs (Donor A)](https://www.10xgenomics.com/resources/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and
can be downloaded in MTX format from [link](https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz)

The cells [annotations](https://github.com/scverse/scanpy_usage/blob/master/170503_zheng17/data/zheng17_bulk_lables.txt) for this dataset can be downloaded from one of the [scanpy_usage](https://github.com/scverse/scanpy_usage) examples for single cell data processing in [scverse](https://github.com/scverse) package.

[Data Approval in Research](https://rdc.apps.res.ibm.com/datasets/1204) for usage this processed data from 10xgenomics

The data is managed on CCC in
```
/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/Zheng68K/
```

## Preprocessing
The python script [zheng68K_convert.py](../../../benchmarks/cell_type_prediction/datasets/Zheng68K/zheng68K_convert.py) was used for converting this data set from MTX to AnnData format.
After this conversion, the annotations of the cell types are managed in observations dataframe (obs) of the anndata object in 'celltype' column


The Zheng68k dataset contains rare cell types, and the distribution of cell types in this dataset is imbalanced. Strong correlations between cell types make it difficult to differentiate them.


## Mapping of Zheng68k cell types to Cell Ontology
Zheng68k cell types where manually mapped to [OLS cell ontology](https://www.ebi.ac.uk/ols4/ontologies/cl/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCL_0000000?lang=en) `cell_type_ontology_term_id` ids. The mapping was done by revieweing te cell type name and description
in https://cellxgene.cziscience.com/collections, and making sure this cell type exists in the cellxgene datasets (which have samples for 580 out of
the possible 2900 possible cell types). The mapping might need furthur review and fine-tuning.

The mapping is stored in [metadata_extra_mapping.csv](/metadata_extra_mapping.csv) which is loaded by the `Zheng68kDataset` object,
to create the `metadata.csv` file which is loaded with the sampels, and so the `cell_type_ontology_term_id`s are available as part of the `sample.metadata` object.
Warning!! `metadata.csv` is kept on the file system between runs and is not generated again if already exists, therefpre it is important to manualy delete it whenever the mapping is updated.
