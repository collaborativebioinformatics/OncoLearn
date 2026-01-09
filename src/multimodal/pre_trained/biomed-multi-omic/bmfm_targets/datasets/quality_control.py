import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from bmfm_targets.datasets.datasets_utils import random_subsampling


def assign_mito_qc_metrics(adata):
    mt_genes = adata.var_names.str.startswith("MT-")

    if mt_genes.any():
        adata.var["mt"] = mt_genes
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    return adata


def transform_X_to_array(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    else:
        adata.X = np.array(adata.X)
    return adata


def remove_outliers(adata):
    """
    Performs outlier removal on 3 QC measures:
    1. The number of counts per cell
    2. The number of genes expressed per cell
    3. The fraction of counts from mitochondrial genes per barcode.

    Outlier removal means removing anything cell that has their value above/below median +- 1.5*IQR.
    """
    adata = transform_X_to_array(adata)

    total_counts_per_barcode = np.sum(adata.X, axis=1)
    total_counts_per_barcode = np.log1p(total_counts_per_barcode)

    num_genes_per_cell = np.sum(adata.X > 0, axis=1)
    num_genes_per_cell = np.log1p(num_genes_per_cell)

    if "pct_counts_mt" in adata.obs.columns:
        pct_mito = adata.obs["pct_counts_mt"]
        pct_mito = np.log1p(pct_mito)

    def iqr_filter(series):
        Q1 = np.percentile(series, 25)
        Q3 = np.percentile(series, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    n_counts_min, n_counts_max = iqr_filter(total_counts_per_barcode)
    n_genes_min, n_genes_max = iqr_filter(num_genes_per_cell)

    if "pct_counts_mt" in adata.obs.columns:
        mito_pct_min, mito_pct_max = iqr_filter(pct_mito)

    # get cells within the acceptable range for all three metrics
    if "pct_counts_mt" in adata.obs.columns:
        valid_cells = (
            (total_counts_per_barcode >= n_counts_min)
            & (total_counts_per_barcode <= n_counts_max)
            & (num_genes_per_cell >= n_genes_min)
            & (num_genes_per_cell <= n_genes_max)
            & (pct_mito >= mito_pct_min)
            & (pct_mito <= mito_pct_max)
        )
    else:
        valid_cells = (
            (total_counts_per_barcode >= n_counts_min)
            & (total_counts_per_barcode <= n_counts_max)
            & (num_genes_per_cell >= n_genes_min)
            & (num_genes_per_cell <= n_genes_max)
        )

    num_cells_removed = np.sum(~valid_cells)
    total_cells = adata.shape[0]
    adata_filtered = adata[valid_cells].copy()
    percentage_removed = (num_cells_removed / total_cells) * 100

    print(f"Number of cells removed: {num_cells_removed}")
    print(f"Percentage of the dataset removed: {percentage_removed:.2f}%")

    return adata_filtered


def add_percentile_lines(ax, data_array):
    data_array = data_array[~np.isnan(data_array)].flatten()
    Q1, median, Q3 = np.percentile(data_array, [25, 50, 75])
    IQR = Q3 - Q1
    lower_bound = median - 1.5 * IQR
    upper_bound = median + 1.5 * IQR
    percentiles = [lower_bound, Q1, median, Q3, upper_bound]
    colors = ["green", "royalblue", "blue", "navy", "red"]
    labels = ["Median - 1.5IQR", "Q1", "Median", "Q3", "Median + 1.5IQR"]
    lines = []

    for percentile, color, label in zip(percentiles, colors, labels):
        line = ax.axvline(
            x=percentile, color=color, linestyle="--", linewidth=1, label=label
        )
        lines.append(line)
    ax.legend(handles=lines, loc="upper right", fontsize="small")


def plot_counts_per_cell_histogram(
    ax,
    adata,
    summary_cells_removed,
    see_percentile_lines=False,
    zoom_plot_kwargs=None,
    cell_count_limits: tuple | None = None,
    log_covariate=False,
):
    adata = transform_X_to_array(adata)
    total_counts_per_barcode = np.sum(adata.X, axis=1)
    if log_covariate:
        total_counts_per_barcode = np.log1p(total_counts_per_barcode)

    if cell_count_limits:
        N, bins, patches = ax.hist(
            total_counts_per_barcode, bins=50, log=True, edgecolor="black"
        )

        min_cell_counts_thres, max_cell_counts_thres = cell_count_limits
        below_min_mask = total_counts_per_barcode < min_cell_counts_thres
        above_max_mask = total_counts_per_barcode > max_cell_counts_thres

        ax.axvline(x=min_cell_counts_thres, color="darkgreen", linewidth=1)
        ax.axvline(x=max_cell_counts_thres, color="red", linewidth=1)

        cells_below = np.sum(below_min_mask)
        cells_above = np.sum(above_max_mask)
        cells_removed = cells_below + cells_above
        cells_removed_bool = below_min_mask | above_max_mask
        summary_cells_removed["cells_removed_count_depth"] = cells_removed_bool
        percent_removed = (cells_removed / adata.shape[0]) * 100

        print(
            f"------ Removing cells according to the count depth -----\n"
            f"Number of cells under threshold: {cells_below}\n"
            f"Number of cells above threshold: {cells_above}\n"
            f"Total number of cells removed: {cells_removed} ({percent_removed:.2f}% of the dataset)\n"
            f"--------------------------------------------------------------"
        )

        for bin_edge, patch in zip(bins, patches):
            if bin_edge < min_cell_counts_thres:
                patch.set_facecolor("lightgrey")
            elif bin_edge > max_cell_counts_thres:
                patch.set_facecolor("lightgrey")
    else:
        ax.hist(total_counts_per_barcode, bins=50, log=True, edgecolor="black")

    if log_covariate:
        ax.set_xlabel("Log of counts per cells", weight="bold")
    else:
        ax.set_xlabel("Counts per cells", weight="bold")
    ax.set_ylabel("Frequency", weight="bold")

    if see_percentile_lines:
        add_percentile_lines(ax, total_counts_per_barcode)
    if zoom_plot_kwargs is not None:
        add_zoomed_image(
            ax, zoom_plot_kwargs, total_counts_per_barcode, covariate="counts_per_cell"
        )


def add_zoomed_image(ax, zoom_plot_kwargs, data_series, covariate):
    if covariate == "counts_per_cell":
        min_covariate_lim = zoom_plot_kwargs.get("min_counts_barcode", 0)
        max_covariate_lim = zoom_plot_kwargs.get("max_counts_barcode")
    elif covariate == "genes_per_cell":
        min_covariate_lim = zoom_plot_kwargs.get("min_genes_per_cell", 0)
        max_covariate_lim = zoom_plot_kwargs.get("max_genes_per_cell")
    elif covariate == "fraction_mito":
        min_covariate_lim = zoom_plot_kwargs.get("min_lim_mt_counts")
        max_covariate_lim = zoom_plot_kwargs.get("max_lim_mt_counts")

    ax_inset = inset_axes(ax, width="50%", height="50%", loc="upper right")
    sns.histplot(data_series, bins=200, legend=False, ax=ax_inset)
    ax_inset.set_xlim(min_covariate_lim, max_covariate_lim)
    ax_inset.tick_params(axis="both", which="major", labelsize=6)
    ax_inset.set_yscale("log")
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")


def plot_cell_rank_vs_counts(ax, adata, cell_count_limits: tuple | None = None):
    total_counts_per_barcode = np.sum(adata.X, axis=1)
    sorted_counts_per_barcode = np.sort(total_counts_per_barcode, axis=0)[::-1]
    x = np.arange(len(sorted_counts_per_barcode))
    ax.plot(x, sorted_counts_per_barcode)
    ax.set_yscale("log")
    ax.set_xlabel("Cell rank", weight="bold")
    ax.set_ylabel("Counts per cell", weight="bold")
    ax.set_xticks(np.linspace(0, max(x), 5))

    if cell_count_limits:
        min_cell_counts_thres = cell_count_limits[0]
        max_cell_counts_thres = cell_count_limits[1]
        ax.axhline(y=min_cell_counts_thres, color="darkgreen", linewidth=1)
        ax.axhline(y=max_cell_counts_thres, color="red", linewidth=1)


def plot_num_genes_per_cell_histogram(
    ax,
    adata,
    summary_cells_removed,
    see_percentile_lines=False,
    zoom_plot_kwargs=None,
    gene_count_limits: tuple | None = None,
    log_covariate=False,
):
    adata = transform_X_to_array(adata)
    num_genes_per_cell = (adata.X > 0).sum(axis=1)
    if log_covariate:
        num_genes_per_cell = np.log1p(num_genes_per_cell)

    if gene_count_limits:
        N, bins, patches = ax.hist(
            num_genes_per_cell, bins=50, log=True, edgecolor="black"
        )

        min_gene_counts_thres, max_gene_counts_thres = gene_count_limits
        below_min_mask = num_genes_per_cell < min_gene_counts_thres
        above_max_mask = num_genes_per_cell > max_gene_counts_thres

        ax.axvline(x=min_gene_counts_thres, color="darkgreen", linewidth=1)
        ax.axvline(x=max_gene_counts_thres, color="red", linewidth=1)

        cells_below = np.sum(below_min_mask)
        cells_above = np.sum(above_max_mask)
        cells_removed = cells_below + cells_above
        cells_removed_bool = below_min_mask | above_max_mask
        summary_cells_removed["cells_removed_genes_expressed"] = cells_removed_bool
        percent_removed = (cells_removed / adata.shape[0]) * 100

        print(
            f"------ Removing cells according to the number of genes expressed -----\n"
            f"Number of cells under threshold: {cells_below}\n"
            f"Number of cells above threshold: {cells_above}\n"
            f"Total number of cells removed: {cells_removed} ({percent_removed:.2f}% of the dataset)\n"
            f"--------------------------------------------------------------------------"
        )

        for bin_edge, patch in zip(bins, patches):
            if bin_edge < min_gene_counts_thres or bin_edge > max_gene_counts_thres:
                patch.set_facecolor("lightgrey")

    else:
        ax.hist(num_genes_per_cell, bins=50, log=True, edgecolor="black")

    if log_covariate:
        ax.set_xlabel("Log of number of genes detected", weight="bold")
    else:
        ax.set_xlabel("Number of genes detected", weight="bold")
    ax.set_ylabel("Frequency", weight="bold")

    if see_percentile_lines:
        add_percentile_lines(ax, num_genes_per_cell)

    if zoom_plot_kwargs is not None:
        add_zoomed_image(
            ax, zoom_plot_kwargs, num_genes_per_cell, covariate="genes_per_cell"
        )


def plot_gene_counts_rank_vs_counts(ax, adata, gene_count_limits: tuple | None = None):
    num_genes_per_cell = (adata.X > 0).sum(axis=1)
    sorted_gene_count_per_cell = np.sort(num_genes_per_cell, axis=0)[::-1]
    x = np.arange(len(sorted_gene_count_per_cell))
    ax.plot(x, sorted_gene_count_per_cell)
    ax.set_yscale("log")
    ax.set_xlabel("Cell rank", weight="bold")
    ax.set_ylabel("Number of Genes Detected", weight="bold")
    ax.set_xticks(np.linspace(0, max(x), 5))

    if gene_count_limits:
        min_gene_counts_thres = gene_count_limits[0]
        max_gene_counts_thres = gene_count_limits[1]
        ax.axhline(y=min_gene_counts_thres, color="darkgreen", linewidth=1)
        ax.axhline(y=max_gene_counts_thres, color="red", linewidth=1)


def plot_mito_fraction_histogram(
    ax,
    adata,
    summary_cells_removed,
    see_percentile_lines=False,
    zoom_plot_kwargs=None,
    mito_fraction_limits: tuple | None = None,
    log_covariate=False,
):
    if "pct_counts_mt" in adata.obs.columns:
        pct_mito = adata.obs["pct_counts_mt"].values

        if log_covariate:
            pct_mito = np.log1p(pct_mito)

        if mito_fraction_limits:
            N, bins, patches = ax.hist(
                pct_mito, bins=50, log=True, edgecolor="black", alpha=0.8
            )

            min_mito_fraction_thres, max_mito_fraction_thres = mito_fraction_limits
            below_min_mask = pct_mito < min_mito_fraction_thres
            above_max_mask = pct_mito > max_mito_fraction_thres

            ax.axvline(x=min_mito_fraction_thres, color="darkgreen", linewidth=1)
            ax.axvline(x=max_mito_fraction_thres, color="red", linewidth=1)

            cells_below = np.sum(below_min_mask)
            cells_above = np.sum(above_max_mask)
            cells_removed = cells_below + cells_above
            cells_removed_bool = below_min_mask | above_max_mask
            summary_cells_removed["cells_removed_mito_genes"] = cells_removed_bool
            percent_removed = (cells_removed / adata.shape[0]) * 100

            print(
                f"------ Removing cells according to mitochondrial genes fraction -----\n"
                f"Number of cells under threshold: {cells_below}\n"
                f"Number of cells above threshold: {cells_above}\n"
                f"Total number of cells removed: {cells_removed} ({percent_removed:.2f}% of the dataset)\n"
                f"--------------------------------------------------------------------------"
            )

            for bin_edge, patch in zip(bins, patches):
                if (
                    bin_edge < min_mito_fraction_thres
                    or bin_edge > max_mito_fraction_thres
                ):
                    patch.set_facecolor("lightgrey")

        else:
            ax.hist(pct_mito, bins=50, log=True, edgecolor="black", alpha=0.8)

        if log_covariate:
            ax.set_xlabel("Log of fraction of mitochondrial counts", weight="bold")
        else:
            ax.set_xlabel("Fraction of mitochondrial counts", weight="bold")
        ax.set_ylabel("Frequency", weight="bold")

        if see_percentile_lines:
            add_percentile_lines(ax, pct_mito)

        if zoom_plot_kwargs is not None:
            add_zoomed_image(ax, zoom_plot_kwargs, pct_mito, covariate="fraction_mito")
    else:
        ax.text(
            0.5, 0.5, "No mitochondrial genes in this dataset", ha="center", va="center"
        )


def plot_mito_vs_genes_scatter(
    ax,
    adata,
    mito_fraction_limits: tuple | None = None,
    gene_count_limits: tuple | None = None,
    cell_count_limits: tuple | None = None,
    log_log_plot=True,
):
    if "pct_counts_mt" in adata.obs.columns:
        pct_mito = adata.obs["pct_counts_mt"].values
        total_counts_per_barcode = np.sum(adata.X, axis=1)
        num_genes_per_cell = (adata.X > 0).sum(axis=1)
        if mito_fraction_limits:
            mito_threshold = mito_fraction_limits[1]
        else:
            mito_threshold = 1  # default threshold

        low_mito_indices = pct_mito <= mito_threshold
        high_mito_indices = pct_mito > mito_threshold

        ax.scatter(
            np.asarray(total_counts_per_barcode).flatten()[low_mito_indices],
            np.asarray(num_genes_per_cell).flatten()[low_mito_indices],
            color="gray",
            alpha=0.5,
            s=5,
        )

        counts_high_mito = np.asarray(total_counts_per_barcode).flatten()[
            high_mito_indices
        ]
        num_genes_high_mito = np.asarray(num_genes_per_cell).flatten()[
            high_mito_indices
        ]
        high_pct_mito_values = pct_mito[high_mito_indices]

        sorted_indices = np.argsort(counts_high_mito)

        total_counts_sorted = counts_high_mito[sorted_indices]
        num_genes_sorted = num_genes_high_mito[sorted_indices]
        pct_mito_sorted = high_pct_mito_values[sorted_indices]

        scatter = ax.scatter(
            total_counts_sorted,
            num_genes_sorted,
            c=pct_mito_sorted,
            cmap="viridis",
            s=5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Percentage mitochondrial counts", weight="bold")
        ax.set_xlabel("Counts per cell", weight="bold")
        ax.set_ylabel("Number of genes detected", weight="bold")

        if log_log_plot:
            ax.set_yscale("log")
            ax.set_xscale("log")

        if gene_count_limits:
            min_gene_counts_thres = gene_count_limits[0]
            ax.axhline(y=min_gene_counts_thres, color="darkgreen", linewidth=1)
        if cell_count_limits:
            min_cell_counts_thres = cell_count_limits[0]
            ax.axvline(x=min_cell_counts_thres, color="darkgreen", linewidth=1)
    else:
        ax.text(
            0.5, 0.5, "No mitochondrial genes in this dataset", ha="center", va="center"
        )


def plot_umap(ax, adata, cell_type_label):
    adata.X = adata.X.astype(float)
    adata_copy = sc.tl.pca(adata, n_comps=20, copy=True)
    sc.pp.neighbors(adata_copy, use_rep="X_pca")
    sc.tl.umap(adata_copy)
    sc.pl.umap(adata_copy, color=cell_type_label, ax=ax)


def show_qc_plots(
    adata,
    cell_type_label,
    mito_fraction_limits: tuple | None = None,
    gene_count_limits: tuple | None = None,
    cell_count_limits: tuple | None = None,
    zoom_plot_kwargs=None,
    see_percentile_lines=False,
):
    import matplotlib.gridspec as gridspec

    summary_cells_removed = {}

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)

    # First row
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("A")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("B")
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("C")
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title("D")

    plot_counts_per_cell_histogram(
        ax1,
        adata,
        summary_cells_removed,
        see_percentile_lines,
        zoom_plot_kwargs,
        cell_count_limits,
    )
    plot_cell_rank_vs_counts(ax2, adata, cell_count_limits)
    plot_num_genes_per_cell_histogram(
        ax3,
        adata,
        summary_cells_removed,
        see_percentile_lines,
        zoom_plot_kwargs,
        gene_count_limits,
    )
    plot_gene_counts_rank_vs_counts(ax4, adata, gene_count_limits)

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_title("E")
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_title("F")
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_title("G")

    plot_mito_fraction_histogram(
        ax5,
        adata,
        summary_cells_removed,
        see_percentile_lines,
        zoom_plot_kwargs,
        mito_fraction_limits,
    )
    plot_mito_vs_genes_scatter(
        ax6, adata, mito_fraction_limits, gene_count_limits, cell_count_limits
    )
    if (
        adata.shape[0] > 70000
    ):  # this is approximately the resources limit to compute pca and knn neighbors
        adata = random_subsampling(adata=adata, n_samples=60000, shuffle=True)
    plot_umap(ax7, adata, cell_type_label)

    if mito_fraction_limits and gene_count_limits and cell_count_limits:
        any_removed = (
            summary_cells_removed["cells_removed_count_depth"]
            | summary_cells_removed["cells_removed_genes_expressed"]
            | summary_cells_removed["cells_removed_mito_genes"]
        )

        total_cells_removed = np.sum(any_removed)
        total_cells_in_dataset = any_removed.shape[0]
        percentage_removed = (total_cells_removed / total_cells_in_dataset) * 100

        print(f"Total number of cells removed: {total_cells_removed}")
        print(f"Percentage of the dataset removed: {percentage_removed:.2f}%")

    plt.tight_layout()
    plt.show()
