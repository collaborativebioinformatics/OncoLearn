#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import numpy as np
import scanpy
from litdata import optimize
from scipy.sparse import csr_matrix


def convert_files(name):
    data = H5ad2LitDataset(name)
    yield from data


class H5ad2LitDataset:
    def __init__(self, name):
        self.name = name
        self.data = scanpy.read_h5ad(self.name)
        self.gene_vocab = np.array(self.data.var_names)

    def __len__(self):
        n = self.data.shape[0]
        return n

    def __iter__(self):
        X = self.data.X
        for idx in range(X.shape[0]):
            start_index = X.indptr[idx]
            end_index = X.indptr[idx + 1]
            nz_rows = X.indices[start_index:end_index]
            expression_values = X.data[start_index:end_index]
            genes = self.gene_vocab[nz_rows].tolist()
            sample = {
                "cell": self.data.obs_names[idx],
                "genes": " ".join(genes),
                "expressions": expression_values,
            }
            yield sample

    def __getstate__(self):
        dict = self.__dict__.copy()
        del dict["data"]
        return dict

    def __setstate__(self, d):
        self.__dict__ = d
        self.data: csr_matrix = scanpy.read_h5ad(self.name)


def convert_h5ad_to_litdata(input_dir, output_dir, num_workers=4, chunk_bytes="128MB"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for split in ("train", "dev", "test"):
        input_names = [i.as_posix() for i in (input_dir / split).glob("*.h5ad")]
        optimize(
            fn=convert_files,
            inputs=input_names,
            output_dir=(output_dir / split).as_posix(),
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        )


def path_type_exists(name):
    path = Path(name)
    if path.exists():
        raise FileExistsError(f"The directory {name} already exists.")
    os.makedirs(name)
    return path


def path_type_not_exists(name):
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"The directory {name} does not exists.")
    return path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=path_type_not_exists,
        help="Input directory with h5ad dataset (train, dev, test).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=path_type_exists,
        help="Output directory for litdata dataset (train, dev, test).",
    )

    parser.add_argument(
        "-nw", "--num_workers", help="Number of workers.", type=int, default=4
    )
    parser.add_argument(
        "-cb",
        "--chunk_bytes",
        help="Size of chunks in bytes (example: 128MB).",
        type=str,
        default="128MB",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    convert_h5ad_to_litdata(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
    )
