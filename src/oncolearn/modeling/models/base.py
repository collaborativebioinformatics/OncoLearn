"""
Base PyTorch Lightning classifier shared by all registered OncoLearn models.
"""
from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

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

    @staticmethod
    def _resolve_class(dotted_name: str):
        """Import and return a class from a dotted module path."""
        mod_path, cls_name = dotted_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod_path), cls_name)

    def __init__(self, config: "OncoLearnConfig") -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(config))
        self._training_cfg = config.training
        self._use_class_weights = config.training.use_class_weights
        self._l1_lambda = config.training.regularization.l1_lambda

        # Build loss function from config
        loss_cfg = config.training.loss
        loss_params = dict(loss_cfg.params)
        if loss_cfg.name == "torch.nn.CrossEntropyLoss":
            ls = config.training.regularization.label_smoothing
            if ls > 0.0:
                loss_params.setdefault("label_smoothing", ls)
        loss_cls = self._resolve_class(loss_cfg.name)
        self.loss_fn = loss_cls(**loss_params)

        num_classes = getattr(config.model, "num_stage_classes", 2)
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        })
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self):
        dm = self.trainer.datamodule
        if (
            self._use_class_weights
            and hasattr(dm, "class_weights")
            and dm.class_weights is not None
        ):
            w = dm.class_weights.to(self.device)
            loss_cfg = self._training_cfg.loss
            loss_params = dict(loss_cfg.params)
            loss_params["weight"] = w
            if loss_cfg.name == "torch.nn.CrossEntropyLoss":
                ls = self._training_cfg.regularization.label_smoothing
                if ls > 0.0:
                    loss_params.setdefault("label_smoothing", ls)
            loss_cls = self._resolve_class(loss_cfg.name)
            self.loss_fn = loss_cls(**loss_params)
            import logging
            logging.getLogger(__name__).info(
                "Class weights applied to loss: %s", w.tolist()
            )

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        labels = batch.get(
            "label",
            torch.zeros(preds["stage_logits"].shape[0], dtype=torch.long, device=self.device),
        )
        loss = self.loss_fn(preds["stage_logits"], labels)
        if self._l1_lambda > 0.0:
            l1 = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
            loss = loss + self._l1_lambda * l1
        self.log("train_loss", loss, prog_bar=True, batch_size=labels.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        preds = self(batch)
        loss = self.loss_fn(preds["stage_logits"], labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=labels.shape[0])
        self.val_metrics.update(preds["stage_logits"], labels)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log("val_acc", metrics["val_acc"], prog_bar=True)
        self.log("val_f1", metrics["val_f1"], prog_bar=True)
        self.log("val_precision", metrics["val_precision"])
        self.log("val_recall", metrics["val_recall"])
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        preds = self(batch)
        loss = self.loss_fn(preds["stage_logits"], labels)
        self.log("test_loss", loss, batch_size=labels.shape[0])
        self.test_metrics.update(preds["stage_logits"], labels)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log("test_acc", metrics["test_acc"])
        self.log("test_f1", metrics["test_f1"])
        self.log("test_precision", metrics["test_precision"])
        self.log("test_recall", metrics["test_recall"])
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt_cfg = self._training_cfg.optimizer
        opt_cls = self._resolve_class(opt_cfg.name)
        optimizer = opt_cls(self.parameters(), **opt_cfg.params)

        sched_cfg = self._training_cfg.scheduler
        sched_params = dict(sched_cfg.params)
        # Auto-fill T_max = max_epochs for CosineAnnealingLR when not set or null
        if sched_cfg.name == "torch.optim.lr_scheduler.CosineAnnealingLR":
            if sched_params.get("T_max") is None:
                t_max = self.trainer.max_epochs if self.trainer else self._training_cfg.max_epochs
                sched_params["T_max"] = t_max
        sched_cls = self._resolve_class(sched_cfg.name)
        scheduler = sched_cls(optimizer, **sched_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": sched_cfg.monitor,
                "interval": sched_cfg.interval,
                "frequency": sched_cfg.frequency,
            },
        }
