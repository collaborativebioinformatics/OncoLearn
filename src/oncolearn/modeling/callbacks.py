"""Custom PyTorch Lightning callbacks for OncoLearn."""
from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl


class MetricsJsonCallback(pl.Callback):
    """Writes per-epoch train (and optionally val) metrics to JSON files.

    Files are written incrementally after each epoch so partial runs are
    recoverable:
      - ``{output_dir}/train_metrics.json``
      - ``{output_dir}/val_metrics.json``  (only when ``log_val=True``)
    """

    def __init__(self, output_dir: str | Path, log_val: bool = True) -> None:
        self._output_dir = Path(output_dir)
        self._log_val = log_val
        self._train_epochs: list = []
        self._val_epochs: list = []

    def _flush(self, path: Path, data: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        row = {"epoch": trainer.current_epoch}
        row.update(
            {k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith("train_")}
        )
        self._train_epochs.append(row)
        self._flush(self._output_dir / "train_metrics.json", self._train_epochs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._log_val:
            return
        row = {"epoch": trainer.current_epoch}
        row.update(
            {k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith("val_")}
        )
        self._val_epochs.append(row)
        self._flush(self._output_dir / "val_metrics.json", self._val_epochs)
