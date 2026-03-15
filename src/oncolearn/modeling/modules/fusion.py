"""
Gated late fusion module: dynamically builds encoders from config.
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig


logger = logging.getLogger(__name__)


def _safe_key(key: str) -> str:
    """Sanitize a dotted batch-routing key for use as an ``nn.ModuleDict`` key.

    ``nn.ModuleDict`` forbids ``"."`` in keys; replace with ``"__"``.
    """
    return key.replace(".", "__")


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
                encoders to build; each encoder resolves its own parameters via
                its registered config class.
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        super().__init__()

        from oncolearn.registry import get_encoder

        # Build encoders from config — each encoder reads its own params via
        # resolve_encoder_config, so only the OncoLearnConfig is needed.
        #
        # _encoder_names stores the *batch routing keys* (dotted or short names).
        # nn.ModuleDict keys must not contain ".", so we sanitize via _safe_key().
        self.encoders = nn.ModuleDict()
        self.encoder_dims: dict[str, int] = {}
        self._encoder_names: list[str] = []

        for enc_cfg in config.model.encoders:
            enc_cls = get_encoder(enc_cfg.name)
            encoder = enc_cls(config)
            # Use the modality name as the batch routing key when set,
            # otherwise fall back to the encoder name.
            batch_key = enc_cfg.modality or enc_cfg.name
            self.encoders[_safe_key(batch_key)] = encoder
            self.encoder_dims[batch_key] = encoder.output_dim
            self._encoder_names.append(batch_key)

        if not self._encoder_names:
            raise ValueError(
                "GatedLateFusionModule requires at least one encoder in config.model.encoders."
            )

        self.num_stage_classes = config.model.num_stage_classes
        self.num_subtype_classes = config.model.num_subtype_classes
        self.has_subtype = self.num_subtype_classes > 0
        dropout = config.model.dropout
        self.modality_dropout_prob = config.model.modality_dropout_prob

        # Per-encoder classification heads (keyed by safe key)
        self.stage_heads = nn.ModuleDict(
            {_safe_key(name): nn.Linear(dim, self.num_stage_classes)
             for name, dim in self.encoder_dims.items()}
        )
        if self.has_subtype:
            self.subtype_heads = nn.ModuleDict(
                {_safe_key(name): nn.Linear(dim, self.num_subtype_classes)
                 for name, dim in self.encoder_dims.items()}
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
        from oncolearn.modeling.encoders.image_encoder import MRMGHierarchicalImageEncoder
        encoder = self.encoders[_safe_key(name)]
        if isinstance(encoder, MRMGHierarchicalImageEncoder):
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
            inputs: Dict mapping batch routing key → input tensor.
                    Missing encoders are treated as unavailable and masked out.
            modality_ids: (B, N) modality IDs for the image encoder (ignored otherwise).

        Returns:
            dict with:
                ``'stage_logits'``: (B, num_stage_classes)
                ``'subtype_logits'``: (B, num_subtype_classes) — only if configured.
        """
        # Stochastic modality dropout (training only) — at least one modality is always kept
        if self.training and self.modality_dropout_prob > 0:
            available = [n for n in self._encoder_names if inputs.get(n) is not None]
            if available:
                keep = {n for n in available if random.random() >= self.modality_dropout_prob}
                if not keep:
                    keep = {random.choice(available)}
                if keep != set(available):
                    inputs = {k: v for k, v in inputs.items() if k in keep}

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

        # Collect per-encoder logits for available modalities
        avail = [(i, name) for i, name in enumerate(self._encoder_names) if name in embeddings]

        # Mask unavailable modalities with -inf before softmax
        mask = torch.zeros(B, len(self._encoder_names), device=ref_device)
        for i, _ in avail:
            mask[:, i] = 1.0

        gate_logits = gate_logits * mask + (1.0 - mask) * (-1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_mods)

        avail_indices = [i for i, _ in avail]
        avail_gate_w = gate_weights[:, avail_indices].unsqueeze(-1)  # (B, num_avail, 1)

        stage_stack = torch.stack(
            [self.stage_heads[_safe_key(name)](embeddings[name]) for _, name in avail], dim=1
        )  # (B, num_avail, C)
        stage_logits = (stage_stack * avail_gate_w).sum(dim=1)  # (B, C)

        result = {"stage_logits": stage_logits}

        if self.has_subtype:
            subtype_stack = torch.stack(
                [self.subtype_heads[_safe_key(name)](embeddings[name]) for _, name in avail], dim=1
            )
            result["subtype_logits"] = (subtype_stack * avail_gate_w).sum(dim=1)

        return result
