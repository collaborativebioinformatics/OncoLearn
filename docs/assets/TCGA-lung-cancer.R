library(TCGAbiolinks)

#set path to download files
#setwd()

# query tcga database
query <- GDCquery(project = c("TCGA-LUSC","TCGA-MESO", "TCGA-LUAD"), 
                  data.category = "Transcriptome Profiling",
                  experimental.strategy = "RNA-Seq",
                  data.type = "Gene Expression Quantification", data.format="TSV",
                  workflow.type = "STAR - Counts") # or use UUIDs

# download using API connection more reliably than gdcdata
GDCdownload(query, method = "api", files.per.chunk = 30)
