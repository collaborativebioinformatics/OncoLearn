import logging
import os

from litdata import StreamingDataset

from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

logger = logging.getLogger(__name__)


class StreamingPanglaoDBDataset(StreamingDataset):
    URL = "https://panglaodb.se/bulk.html"

    DATASET_NAME = "PanglaoDB"

    def __init__(
        self,
        input_dir: str,
        split: str,
        shuffle: bool = False,
        drop_last: bool | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
    ):
        # input_dir = "local:" + os.path.abspath(input_dir)
        input_dir = os.path.join(input_dir, split)
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        obj = super().__getitem__(idx)
        return MultiFieldInstance(
            metadata={"cell_name": obj["cell"]},
            data={
                "genes": obj["genes"].split(),
                "expressions": list(obj["expressions"]),
            },
        )


class StreamingPanglaoDBDataModule(StreamingDataModule):
    """PyTorch Lightning DataModule for PanglaoDB dataset."""

    DATASET_FACTORY: StreamingDataset = StreamingPanglaoDBDataset
