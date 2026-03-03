"""
Gated late fusion module: dynamically builds encoders from config.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig


logger = logging.getLogger(__name__)


class GatedLateFusionModule(nn.Module):
    """
    Gated late fusion with per-encoder heads and a learned gating network.

    Encoders are instantiated dynamically from ``config.model.encoders`` via the
    encoder registry, so the set of active modalities is fully config-driven.

    Architecture:
    - Each encoder produces a fixed-size embedding from its input.
    - Per-encoder classification heads produce task-specific logits.
    - A gate network sees the concatenation of all encoder embeddings (zeros for
      missing modalities) and outputs soft attention weights.
    - Final logits = gated weighted sum over per-encoder logits.

    Args:
        config: Full experiment config.  ``config.model.encoders`` specifies which
                encoders to build and their output dimensions.
        device: Optional device for encoder construction (passed to encoders that
                need it, e.g. RNA BERT).
    """

    def __init__(self, config: "OncoLearnConfig", device: torch.device = None) -> None:
        super().__init__()

        from oncolearn.registry import get_encoder

        device_str = str(device) if device else None

        # Build encoders from config
        self.encoders = nn.ModuleDict()
        self.encoder_dims: dict[str, int] = {}
        self._encoder_names: list[str] = []

        for enc_cfg in config.model.encoders:
            enc_cls = get_encoder(enc_cfg.name)
            kwargs = dict(enc_cfg.kwargs)
            if enc_cfg.name == "gene" and device_str:
                kwargs["device"] = device_str
            encoder = enc_cls(config, output_dim=enc_cfg.output_dim, **kwargs)
            self.encoders[enc_cfg.name] = encoder
            self.encoder_dims[enc_cfg.name] = enc_cfg.output_dim
            self._encoder_names.append(enc_cfg.name)

        if not self._encoder_names:
            raise ValueError(
                "GatedLateFusionModule requires at least one encoder in config.model.encoders."
            )

        self.num_stage_classes = config.model.num_stage_classes
        self.num_subtype_classes = config.model.num_subtype_classes
        self.has_subtype = self.num_subtype_classes > 0
        dropout = config.model.dropout

        # Per-encoder classification heads
        self.stage_heads = nn.ModuleDict(
            {name: nn.Linear(dim, self.num_stage_classes) for name, dim in self.encoder_dims.items()}
        )
        if self.has_subtype:
            self.subtype_heads = nn.ModuleDict(
                {name: nn.Linear(dim, self.num_subtype_classes) for name, dim in self.encoder_dims.items()}
            )

        # Gate network: sees all embeddings concatenated (zeros for missing modalities)
        gate_input_dim = sum(self.encoder_dims.values())
        num_mods = len(self._encoder_names)
        hidden_dim = max(64, 64 * num_mods)
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_mods),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self, name: str, data: torch.Tensor, modality_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Dispatch to the correct encoder, passing extra args when needed."""
        encoder = self.encoders[name]
        if name == "image":
            return encoder(data, modality_ids)
        return encoder(data)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: dict,
        modality_ids: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            inputs: Dict mapping encoder name → input tensor.
                    Missing encoders are treated as unavailable and masked out.
            modality_ids: (B, N) modality IDs for the image encoder (ignored otherwise).

        Returns:
            dict with:
                ``'stage_logits'``: (B, num_stage_classes)
                ``'subtype_logits'``: (B, num_subtype_classes) — only if configured.
        """
        embeddings: dict[str, torch.Tensor] = {}
        B = None

        for name in self._encoder_names:
            data = inputs.get(name)
            if data is not None:
                if B is None:
                    B = data.shape[0]
                embeddings[name] = self._encode(name, data, modality_ids)

        if not embeddings:
            raise ValueError("At least one modality input must be provided.")

        # Validate consistent batch sizes
        sizes = {name: emb.shape[0] for name, emb in embeddings.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(f"Inconsistent batch sizes across modalities: {sizes}")

        ref_device = next(iter(embeddings.values())).device

        # Build gate input (zero-fill for unavailable modalities)
        gate_parts = [
            embeddings[name]
            if name in embeddings
            else torch.zeros(B, self.encoder_dims[name], device=ref_device)
            for name in self._encoder_names
        ]
        gate_logits = self.gate_network(torch.cat(gate_parts, dim=-1))  # (B, num_mods)

        # Mask unavailable modalities with -inf before softmax
        mask = torch.zeros(B, len(self._encoder_names), device=ref_device)
        avail_indices = []
        for i, name in enumerate(self._encoder_names):
            if name in embeddings:
                mask[:, i] = 1.0
                avail_indices.append(i)

        gate_logits = gate_logits * mask + (1.0 - mask) * (-1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_mods)

        # Collect per-encoder logits for available modalities
        avail = [(i, name) for i, name in enumerate(self._encoder_names) if name in embeddings]
        avail_gate_w = gate_weights[:, avail_indices].unsqueeze(-1)  # (B, num_avail, 1)

        stage_stack = torch.stack(
            [self.stage_heads[name](embeddings[name]) for _, name in avail], dim=1
        )  # (B, num_avail, C)
        stage_logits = (stage_stack * avail_gate_w).sum(dim=1)  # (B, C)

        result = {"stage_logits": stage_logits}

        if self.has_subtype:
            subtype_stack = torch.stack(
                [self.subtype_heads[name](embeddings[name]) for _, name in avail], dim=1
            )
            result["subtype_logits"] = (subtype_stack * avail_gate_w).sum(dim=1)

        return result
