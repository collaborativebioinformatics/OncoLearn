library(tidyverse)
library(TCGAbiolinks)
library(GEOquery)
library(SummarizedExperiment)

# Set output directory for lung cancer data
data_dir <- "./data"

# Create directory if it doesn't exist and set as working directory
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
setwd(data_dir)

# Query TCGA database for lung cancer cohorts (LUSC, MESO, LUAD)
query <- GDCquery(project = c("TCGA-LUSC", "TCGA-MESO", "TCGA-LUAD"),
                  data.category = "Transcriptome Profiling",
                  experimental.strategy = "RNA-Seq",
                  data.type = "Gene Expression Quantification",
                  data.format = "TSV",
                  workflow.type = "STAR - Counts")

# Download data using API connection
GDCdownload(query, method = "api", files.per.chunk = 30)

# Prepare and save the RNA-Seq data
output_filename <- "TCGA_Lung-RNASeq.RData"
data <- GDCprepare(query = query,
                   save.filename = output_filename,
                   save = TRUE)