import datetime
import logging
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import rdata
from anndata import AnnData
from scipy import sparse
from sklearn.utils import shuffle

logging.basicConfig(
    level=logging.INFO,
    filename="panglaodb_dataset.log",
    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sra_splits(
    data_info_path: str | Path,
    rdata_dir: str | Path,
    filter_query: str | None = None,
    random_state=42,
) -> dict[str, pd.DataFrame]:
    """
    Create train, dev, and test splits based on the provided filter query.

    Args:
    ----
        data_info_path (str | Path): Path to the data info file.
        rdata_dir (str | Path): Path to the directory containing the RData files.
        filter_query (str | None, optional): Filter query to apply to the data info file. Defaults to None.
        random_state (int, optional): Random state for shuffling the data. Defaults to 42.

    Returns:
    -------
        dict: Dictionary containing the splits as keys and corresponding dataframes as values.

    Raises:
    ------
        FileNotFoundError: If the data info file is not found.

    Note:
    ----
        This method assumes that the data info file is in CSV format with columns:
        'SRA Accession', 'SRS Accession', and 'file'.
    """
    if not os.path.exists(data_info_path):
        raise FileNotFoundError(f"Data info file '{data_info_path}' not found.")

    sra_df = pd.read_csv(data_info_path)
    sra_df = sra_df.drop(sra_df[sra_df["SRS Accession"] == "notused"].index)
    sra_df["sra_location"] = sra_df.apply(
        lambda x: os.path.join(
            rdata_dir,
            f"{x['SRA Accession']}_{x['SRS Accession']}.sparse.RData",
        ),
        axis=1,
    )

    if filter_query is not None:
        sra_df = sra_df.query(filter_query)

    shuffled_df = shuffle(sra_df, random_state=random_state)

    total_rows = len(shuffled_df)
    train_rows = int(0.8 * total_rows)
    dev_rows = int(0.1 * total_rows)

    train_df = shuffled_df[:train_rows]
    dev_df = shuffled_df[train_rows : train_rows + dev_rows]
    test_df = shuffled_df[train_rows + dev_rows :]

    return {"train": train_df, "dev": dev_df, "test": test_df}


def convert_all_rdatas_to_h5ad(h5ad_dir, sra_df_dict, num_workers=None) -> None:
    """
    Convert RData files to h5ad format for all datasets in the splits.

    Args:
    ----
        h5ad_dir (str | Path): Path to the output directory.
        sra_df_dict (dict): Dictionary containing the splits as keys and corresponding dataframes as values.
        num_workers (int, optional): Number of workers for multiprocessing. Defaults to None.


    Note:
    ----
        This method assumes that the 'h5ad_dir' attribute is set to the desired output directory.
        The 'sra_df_dict' attribute should contain the splits ('train', 'dev', 'test') as keys and
        corresponding dataframes as values, each containing 'sra_location' column with the RData file paths.
    """
    if not os.path.exists(h5ad_dir):
        os.makedirs(h5ad_dir)

    for split, sra_df in sra_df_dict.items():
        if sra_df.empty:
            logger.warning(f'Split "{split}" empty! Skipping...')
            continue
        sra_locations = sra_df["sra_location"].to_numpy().tolist()
        sra_study_names = sra_df.apply(
            lambda row: row["SRA Accession"] + "_" + row["SRS Accession"], axis=1
        ).tolist()
        split_dir = os.path.join(h5ad_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        h5ad_locations = [
            os.path.join(split_dir, sra_study_name + ".h5ad")
            for sra_study_name in sra_study_names
        ]
        logger.info(f"Number of datasets: {len(sra_locations)}")
        if num_workers != 0:
            pool = multiprocessing.Pool(processes=num_workers)
            pool.starmap(
                _convert_rdata_to_h5ad,
                zip(sra_locations, h5ad_locations),
            )
        else:
            for sra_loc, h5ad_loc in zip(sra_locations, h5ad_locations):
                _convert_rdata_to_h5ad(sra_loc, h5ad_loc)


def _convert_rdata_to_h5ad(sra_location: str, h5ad_location: str) -> None:
    """
    Convert a single RData file to h5ad format.

    Args:
    ----
        sra_location (str): Path to the input RData file.
        h5ad_location (str): Path to the output h5ad file.

    Note:
    ----
        This method uses rdata package for parsing and converting RData files to AnnData h5ad format.
    """
    logger.info(
        "Processing file and starting time: {} {} ".format(
            sra_location, datetime.datetime.now().strftime("%H:%M:%S")
        )
    )
    parsed_file = rdata.parser.parse_file(sra_location)
    converted_data = rdata.conversion.convert(parsed_file)
    if "sm" in converted_data:
        converted_data = converted_data["sm"]
    else:
        converted_data = converted_data["sm2"]
    counts_matrix = sparse.csc_matrix(
        (converted_data.x, converted_data.i, converted_data.p),
        tuple(converted_data.Dim),
    )
    counts_matrix = counts_matrix.transpose()
    study_data = AnnData(counts_matrix)
    study_data.var_names, study_data.obs_names = converted_data.Dimnames

    logger.info(
        "Shape of the dataset, shape of observations, and shape of var names: "
        f"{study_data.X.shape} {len(study_data.obs_names)} {len(study_data.var_names)}"
    )

    logger.info(
        "Processed file and ending time {} {} ".format(
            sra_location, datetime.datetime.now().strftime("%H:%M:%S")
        )
    )
    study_data.write_h5ad(h5ad_location)
