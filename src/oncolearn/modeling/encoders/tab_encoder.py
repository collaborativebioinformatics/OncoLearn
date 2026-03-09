"""
FT-Transformer encoder for tabular data, and MLP encoder for clinical features.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer

from oncolearn.registry import register_config, register_encoder
from .base import BaseEncoder

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


@register_config("oncolearn.encoder.multimodal.FTTransformerEncoder")
@dataclass
class FTTransformerEncoderConfig:
    """Configuration for the FT-Transformer tabular encoder."""

    output_dim: int = 128
    input_dim: int = 32
    dim: int = 128
    num_heads: int = 4
    depth: int = 2
    dropout: float = 0.2


@register_encoder("oncolearn.encoder.multimodal.FTTransformerEncoder")
class FTTransformerEncoder(BaseEncoder):
    """
    FTTransformer encoder for continuous tabular / clinical features.

    Args:
        config: Full experiment config.  Encoder-specific parameters are read via
                :func:`~oncolearn.registry.resolve_encoder_config` which merges
                :class:`FTTransformerEncoderConfig` defaults with any YAML overrides.
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        from oncolearn.registry import resolve_encoder_config

        enc_cfg: FTTransformerEncoderConfig = resolve_encoder_config(
            type(self), "clinical", config
        )

        super().__init__(config, output_dim=enc_cfg.output_dim, huggingface_models=None)
        self.input_dim = enc_cfg.input_dim

        # Project variable-length input features to the fixed input_dim expected by
        # FTTransformer.  LazyLinear defers weight init to the first forward pass so
        # we don't need to know the feature count at construction time.
        self.input_proj = nn.Sequential(
            nn.LazyLinear(enc_cfg.input_dim),
            nn.LayerNorm(enc_cfg.input_dim),
        )

        self.tab_transformer = FTTransformer(
            categories=(),
            num_continuous=enc_cfg.input_dim,
            dim=enc_cfg.dim,
            depth=enc_cfg.depth,
            heads=enc_cfg.num_heads,
            attn_dropout=enc_cfg.dropout,
            ff_dropout=enc_cfg.dropout,
            dim_out=enc_cfg.dim,
        )

        # NOTE: FTTransformer is always trained — it has no pretrained checkpoint so
        # freezing random weights would make the clinical encoder unlearnable.
        if self.freeze_encoders:
            logger.info(
                "freeze_encoders=True but FTTransformerEncoder has no pretrained checkpoint; "
                "training the tab_transformer regardless."
            )

        self.output_proj = (
            nn.Linear(enc_cfg.dim, enc_cfg.output_dim)
            if enc_cfg.dim != enc_cfg.output_dim
            else nn.Identity()
        )
        self.output_norm = nn.LayerNorm(enc_cfg.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x)
        x_categ = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        encoded = self.tab_transformer(x_categ, x)

        return self.output_norm(self.output_proj(encoded))
