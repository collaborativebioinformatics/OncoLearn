# Takes care of missing values, normalizes the data and feature selection using variance
# Filtering - keep genes expressed (>0) in >= fraction of samples
# remove samples with extreme median expression
# keep top genes by variance after scaling
# removes genes with no expression

preprocess_expr <- function(
    expr_df,              
    min_nonzero_frac = 0.10,  
    outlier_z = 4,            
    top_n_var = 2000,         
    pseudocount = 1
) {
  # ---- convert to numeric matrix ----
  X <- as.data.frame(lapply(expr_df, function(z) as.numeric(as.character(z))))
  X <- as.matrix(X)
  
  # ---- handle missing values (median per gene) ----
  for (j in seq_len(ncol(X))) {
    if (anyNA(X[, j])) {
      X[is.na(X[, j]), j] <- stats::median(X[, j], na.rm = TRUE)
    }
  }
  
  # ---- remove genes with mostly zero expression ----
  nonzero_frac <- colMeans(X > 0)
  keep_genes <- nonzero_frac >= min_nonzero_frac
  X <- X[, keep_genes, drop = FALSE]
  
  if (ncol(X) == 0) {
    stop("All genes filtered out. Lower min_nonzero_frac.")
  }
  
  # ---- remove outlier samples ----
  # based on sample-wise/subtype-wise median expression
  s_med <- apply(X, 1, stats::median)
  z <- (s_med - stats::median(s_med)) / stats::mad(s_med, constant = 1)
  outliers <- which(abs(z) > outlier_z)
  
  if (length(outliers) > 0) {
    X <- X[-outliers, , drop = FALSE]
  }
  
  # ---- normalization (log2) ----
  X <- log2(X + pseudocount)
  
  # ---- scaling (z-score per gene) ----
  X <- scale(X)
  
  # ---- feature selection (variance filter) ----
  if (!is.null(top_n_var) && top_n_var < ncol(X)) {
    v <- apply(X, 2, stats::var)
    keep <- names(sort(v, decreasing = TRUE))[seq_len(top_n_var)]
    X <- X[, keep, drop = FALSE]
  }
  
  list(
    X = X,
    removed_outlier_idx = outliers,
    kept_genes = colnames(X)
  )
}


# EXAMPLE USE -- expr_df: subtypes × genes
#expr_df <- cancer_data[, grep("^gene_", names(cancer_data))]
#res <- preprocess_expr(
#  expr_df = expr_df,
#  min_nonzero_frac = 0.10,
#  outlier_z = 4,
#  top_n_var = 2000,
#  pseudocount = 1
#)
#X <- res$X