"""
FT-Transformer encoder for clinical/tabular data (B1).
"""
import logging
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

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

        self.tab_transformer = TabTransformer(
            categories={}, 
            num_continuous=input_dim,
            dim=dim,
            depth=depth,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            mlp_act=nn.GELU(),
            mlp_hidden_dims=[dim * 2, dim],
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

        x_cont = x  # (B, input_dim)
        
        with torch.no_grad():  
            encoded = self.tab_transformer(x_cat=None, x_cont=x_cont)

        output = self.output_proj(encoded) 
        
        return output
