import argparse
from dataclasses import dataclass

from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer


@dataclass
class MultifiledTokenizerConverterConfigSchema:
    # target TokenizerConfig
    input: str
    # target a DataModule
    output: str | None = None
    # list of targets to FieldInfo
    write_back: bool = False

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(
            description="Convert old multifiled tokenizers (in the form of a multifield vocab) to the new format."
        )
        parser.add_argument(
            "--input", "-i", type=str, help="Path of old-format MultiField tokenizer."
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Output directory for converted MultiField tokenizer.",
        )
        parser.add_argument(
            "--write-back",
            action="store_true",
            help="Write converted tokenizer back to the same path where the old tokenizer was found.  Does not override old parser.",
        )
        args = parser.parse_args()
        if not bool(args.output) ^ args.write_back:
            parser.error(
                "You must specify either --output or --write_back but not both."
            )
        return MultifiledTokenizerConverterConfigSchema(**(args.__dict__))


def main():
    cfg = MultifiledTokenizerConverterConfigSchema.from_args()
    print(f"reading old tokenizer from {cfg.input}")
    if cfg.write_back:
        print("Saving a modified tokenizer back to the input path")
    elif cfg.output:
        print(f"Saving a modified tokenizer to {cfg.output}")

    tokenizer = MultiFieldTokenizer.from_old_multifield_tokenizer(
        name_or_path=cfg.input, save_converted_tokenizer_back=cfg.write_back
    )

    if cfg.output:
        tokenizer.save_pretrained(cfg.output)


if __name__ == "__main__":
    main()
