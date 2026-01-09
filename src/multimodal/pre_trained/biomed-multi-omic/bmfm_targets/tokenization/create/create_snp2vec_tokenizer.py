import argparse
import json
import logging
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import (
    AddedToken,
    SentencePieceBPETokenizer,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast


class SNPdbParquetDataset:
    """

    Dataset class to read a parquet file and yield tokenized DNA sequences.

    Args:
    ----
        input_dir (str): Path to the directory containing the parquet files.
    """

    def __init__(self, input_dir, batch_size=1000, limit=None):
        """
        Initialize the dataset.

        Args:
        ----
            input_dir (str): Path to the directory containing the parquet files.
            batch_size (int): Batch size to use.
        """
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.limit = limit

    def __iter__(self):
        file_paths = list(Path(self.input_dir).rglob("*.parquet"))
        if self.limit:
            file_paths = random.sample(file_paths, int(self.limit * len(file_paths)))
        else:
            file_paths = file_paths
        for file_path in file_paths:
            data = pq.ParquetFile(file_path)
            for batch in data.iter_batches(
                batch_size=self.batch_size, use_threads=False, columns=["dna_sequence"]
            ):
                yield [item.strip() for item in batch["dna_sequence"].tolist()]


def train_tokenizer(args):
    """
    Train a tokenizer on the snp dataset.

    Args:
    ----
        args (argparse.Namespace): Command-line arguments.
    """
    logging.info("Training the tokenizer.")
    sp_tokenizer = SentencePieceBPETokenizer(unk_token="[UNK]", add_prefix_space=False)
    sp_tokenizer.decoder = None
    sp_tokenizer.normalizer = None
    sp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = [
        AddedToken(token) for token in ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    ]
    parquet_dataset = SNPdbParquetDataset(
        args.input_dir, args.batch_size, args.limit_percentage
    )
    sp_tokenizer.train_from_iterator(
        iterator=parquet_dataset,
        vocab_size=args.vocab_size,
        min_frequency=10,
        special_tokens=special_tokens,
        limit_alphabet=args.limit_alphabet,
    )
    cls_token_id = sp_tokenizer.token_to_id("[CLS]")
    sep_token_id = sp_tokenizer.token_to_id("[SEP]")
    sp_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=sp_tokenizer)
    fast_tokenizer.save_pretrained(args.output_dir)
    logging.info("Tokenizer training complete!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sentence tokenizer on the snp dataset."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the parquet files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where the tokenizer will be stored.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size during training.",
    )
    parser.add_argument(
        "-vs",
        "--vocab_size",
        type=int,
        default=4096,
        help="Size of the desired vocabulary.",
    )
    parser.add_argument(
        "--limit_alphabet",
        type=int,
        default=1000,
        help="Limit of alphabet.",
    )
    parser.add_argument(
        "-l",
        "--limit_percentage",
        default=0.01,
        type=float,
        help="percentage of files to use for training.",
    )
    args = parser.parse_args()
    return args


def create_multifield_vocab(args):
    d_tokenizer = json.load(open(os.path.join(args.output_dir, "tokenizer.json")))
    d_multifield_vocab = {"dna_chunks": list(d_tokenizer["model"]["vocab"].keys())}
    with open(
        os.path.join(Path(args.output_dir).parent, "multifield_vocab.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(d_multifield_vocab, f, ensure_ascii=False, indent=4)


def main():
    logging.info("Training the tokenizer.")
    args = parse_args()
    train_tokenizer(args)
    # create_multifield_vocab(args)


if __name__ == "__main__":
    main()
