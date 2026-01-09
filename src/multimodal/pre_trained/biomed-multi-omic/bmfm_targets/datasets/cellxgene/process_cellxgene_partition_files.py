import os

import anndata as ad
from scanpy import read_h5ad


def preprocess_cellxgene_partition_file(h5ad_file):
    processed_file = h5ad_file
    # Fix obs object index
    processed_file.obs.index = (
        processed_file.obs["tissue"].astype(str)
        + "_"
        + processed_file.obs["soma_joinid"].astype(str)
    ).str.replace(" ", "_")
    processed_file.obs.index.names = ["index"]
    # Rename genes
    processed_file.var_names = processed_file.var["feature_name"]
    return processed_file


def concatenate_cellxgene_partitioned_files(files_path, n_files):
    all_files = os.listdir(files_path)
    if n_files is not None:
        all_files = all_files[0:n_files]

    concat_file = None
    for i, file in enumerate(all_files):
        print(f"file name: {file}")
        print(f"{i + 1} out of {len(all_files)}")
        current_file = read_h5ad(files_path + file)
        processed_file = preprocess_cellxgene_partition_file(current_file)

        if concat_file is None:
            concat_file = processed_file
        else:
            concat_file = ad.concat([concat_file, processed_file])

    concat_file.obs_names_make_unique()
    return concat_file


def filter_anndata(adata, filter_terms):
    """
    Filter an AnnData object based on the conditions specified in the filter_terms dictionary.

    Parameters
    ----------
        adata (anndata.AnnData): An AnnData object to be filtered.
        filter_terms (dict): A dictionary containing filter conditions.

    Returns
    -------
        anndata.AnnData: A filtered AnnData object.
    """
    filtered_indices = None

    for key, value in filter_terms.items():
        if key not in adata.obs.columns:
            raise ValueError(f"Column '{key}' not found in AnnData object.")

        if isinstance(value, set):
            condition = adata.obs[key].isin(value)
        else:
            condition = adata.obs[key] == value

        if filtered_indices is None:
            filtered_indices = condition
        else:
            filtered_indices = filtered_indices & condition

    filtered_adata = adata[filtered_indices, :]

    return filtered_adata


if __name__ == "__main__":
    dataset_path = (
        "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/"
    )
    anndata_files = "anndata_all/all_unique_cells/"
    files_path = dataset_path + anndata_files
    cellxgene = concatenate_cellxgene_partitioned_files(files_path, n_files=2)
    filter_terms = {"suspension_type": "cell"}
    filtered_cellxgene = filter_anndata(adata=cellxgene, filter_terms=filter_terms)
    save_dir = dataset_path + "h5ad/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filtered_cellxgene.write(save_dir + "cellxgene.h5ad")
