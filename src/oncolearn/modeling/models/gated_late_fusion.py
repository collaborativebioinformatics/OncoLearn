"""
Registered Lightning classifier: Gated Late Fusion.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from oncolearn.registry import register_model
from oncolearn.modeling.models.base import BaseOncoClassifier
from oncolearn.modeling.modules.fusion import GatedLateFusionModule

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig


@register_model("gated_late_fusion", modalities=["tabular", "image"])
class GatedLateFusionClassifier(BaseOncoClassifier):
    """
    PyTorch Lightning classifier wrapping :class:`GatedLateFusionModule`.

    The full experiment config is passed directly to ``__init__``;
    encoder construction is delegated to :class:`GatedLateFusionModule` via the
    encoder registry.  The set of active encoders is fully driven by
    ``config.model.encoders``.
    """

    def __init__(self, config: "OncoLearnConfig", device: torch.device = None) -> None:
        super().__init__(config)
        self._encoder_names = [ec.name for ec in config.model.encoders]
        self.model = GatedLateFusionModule(config, device=device)

    def forward(self, batch):
        inputs = {}
        for name in self._encoder_names:
            # Allow "gene" encoder to receive data from either "gene" or "tabular" batch key.
            tensor = batch.get(name)
            if tensor is None and name == "gene":
                tensor = batch.get("tabular")
            if tensor is not None:
                inputs[name] = tensor

        return self.model(inputs, modality_ids=batch.get("modality_ids"))
