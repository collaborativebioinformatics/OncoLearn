from pathlib import Path

import pandas as pd


def load_split_dataset_ids(split_file: str | Path, split: str = "train") -> list[str]:
    """
    Load dataset_ids from a split file.

    File must be a csv with a column `dataset_id` and a column `split`.

    Args:
    ----
        split_file (str | Path): path to split file
        split (str, optional): desired split. Defaults to "train".

    Returns:
    -------
        list[str]: list of dataset_ids in split
    """
    dataset_splits = pd.read_csv(split_file)
    return [*dataset_splits[dataset_splits.split == split].dataset_id]
