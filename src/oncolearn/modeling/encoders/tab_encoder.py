"""
FT-Transformer encoder for clinical/tabular data.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer

from oncolearn.registry import register_encoder
from .base import BaseEncoder

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


@register_encoder("tabular")
class FTTransformerEncoder(BaseEncoder):
    """
    FTTransformer encoder for continuous tabular / clinical features.

    Args:
        config: Full experiment config. Reads ``model.dropout`` for the dropout rate.
        output_dim: Embedding dimension produced by this encoder.
        input_dim: Number of continuous input features.
        dim: Internal transformer dimension.
        num_heads: Number of attention heads.
        depth: Number of transformer layers.
    """

    def __init__(
        self,
        config: "OncoLearnConfig",
        output_dim: int = 128,
        input_dim: int = 1,
        dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        **kwargs,
    ):
        super().__init__(config, output_dim=output_dim, huggingface_models=None, **kwargs)
        self.input_dim = input_dim

        dropout = config.model.dropout

        self.tab_transformer = FTTransformer(
            categories=(),
            num_continuous=input_dim,
            dim=dim,
            depth=depth,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            dim_out=dim,
        )

        self._freeze(self.tab_transformer)
        logger.info(f"TabTransformer encoder frozen with {input_dim} continuous features")

        self.output_proj = nn.Linear(dim, output_dim) if dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_categ = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        with torch.no_grad():
            encoded = self.tab_transformer(x_categ, x)

        return self.output_proj(encoded)
