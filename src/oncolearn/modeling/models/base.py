"""
Base PyTorch Lightning classifier shared by all registered OncoLearn models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import pytorch_lightning as pl

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig


class BaseOncoClassifier(pl.LightningModule):
    """
    Base class for all registered OncoLearn classifiers.

    Provides shared training / validation / test steps and optimizer
    configuration driven by the experiment config.

    Subclasses must:
      1. Call ``super().__init__(config)``
      2. Assign the underlying ``nn.Module`` to ``self.model``
      3. Implement ``forward(batch) -> dict`` with at least ``'stage_logits'``
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        super().__init__()
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        labels = batch.get(
            "label",
            torch.zeros(preds["stage_logits"].shape[0], dtype=torch.long, device=self.device),
        )
        loss = self.loss_fn(preds["stage_logits"], labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        preds = self(batch)
        loss = self.loss_fn(preds["stage_logits"], labels)
        acc = (preds["stage_logits"].argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        t_max = self.trainer.max_epochs if self.trainer else 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
