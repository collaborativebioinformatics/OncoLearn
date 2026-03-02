"""
FT-Transformer encoder for clinical/tabular data (B1).
"""
import logging
import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer

logger = logging.getLogger(__name__)


class FTTransformerEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.2,
        output_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.output_dim = output_dim

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

        # Freeze TabTransformer parameters
        for param in self.tab_transformer.parameters():
            param.requires_grad = False

        logger.info(f"TabTransformer encoder frozen with {input_dim} continuous features")

        tab_output_dim = dim

        if tab_output_dim != output_dim:
            self.output_proj = nn.Linear(tab_output_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.shape[0]
        x_categ = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        with torch.no_grad():
            encoded = self.tab_transformer(x_categ, x)

        output = self.output_proj(encoded) 
        
        return output
