"""
Registered Lightning classifier: Gated Late Fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from oncolearn.registry import register_config, register_model
from oncolearn.modeling.models.base import BaseOncoClassifier
from oncolearn.modeling.modules.fusion import GatedLateFusionModule

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig


@register_config("oncolearn.model.multimodal.gated_late_fusion")
@dataclass
class GatedLateFusionConfig:
    """Configuration for the Gated Late Fusion model."""

    num_stage_classes: int = 5
    num_subtype_classes: int = 0
    freeze_encoders: bool = True
    dropout: float = 0.2
    modality_dropout_prob: float = 0.0


@register_model(
    "oncolearn.model.multimodal.gated_late_fusion",
    modalities=[
        "oncolearn.modality.gene",
        "oncolearn.modality.clinical",
        "oncolearn.modality.image"
    ]
)
class GatedLateFusionClassifier(BaseOncoClassifier):
    """
    PyTorch Lightning classifier wrapping :class:`GatedLateFusionModule`.

    The full experiment config is passed directly to ``__init__``;
    encoder construction is delegated to :class:`GatedLateFusionModule` via the
    encoder registry.  The set of active encoders is fully driven by
    ``config.model.encoders``.
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        super().__init__(config)
        self.model = GatedLateFusionModule(config)

    def forward(self, batch):
        inputs = {
            key: batch[key]
            for key in self.model._encoder_names
            if key in batch
        }
        return self.model(inputs, modality_ids=batch.get("modality_ids"))
