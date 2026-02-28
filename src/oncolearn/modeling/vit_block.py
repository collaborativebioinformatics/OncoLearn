# """
# Transformer block for Vision Transformer.
# """
# import torch
# import torch.nn as nn


# class TransformerBlock(nn.Module):
#     """Standard transformer block with self-attention and MLP."""
    
#     def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = MultiHeadAttention(dim, num_heads, dropout)
#         self.norm2 = nn.LayerNorm(dim)
        
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Pre-norm architecture
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x
    
#     def load_from_state_dict(self, state_dict: dict, prefix: str):
#         """Load weights from state_dict with prefix."""
#         # Load norm1
#         if f'{prefix}norm1.weight' in state_dict:
#             self.norm1.weight.data = state_dict[f'{prefix}norm1.weight']
#             self.norm1.bias.data = state_dict[f'{prefix}norm1.bias']
        
#         # Load attention
#         self.attn.load_from_state_dict(state_dict, prefix + 'attn.')
        
#         # Load norm2
#         if f'{prefix}norm2.weight' in state_dict:
#             self.norm2.weight.data = state_dict[f'{prefix}norm2.weight']
#             self.norm2.bias.data = state_dict[f'{prefix}norm2.bias']
        
#         # Load MLP
#         if f'{prefix}mlp.fc1.weight' in state_dict:
#             self.mlp[0].weight.data = state_dict[f'{prefix}mlp.fc1.weight']
#             self.mlp[0].bias.data = state_dict[f'{prefix}mlp.fc1.bias']
#         if f'{prefix}mlp.fc2.weight' in state_dict:
#             self.mlp[3].weight.data = state_dict[f'{prefix}mlp.fc2.weight']
#             self.mlp[3].bias.data = state_dict[f'{prefix}mlp.fc2.bias']


# class MultiHeadAttention(nn.Module):
#     """Multi-head self-attention."""
    
#     def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # QKV projection (combined for efficiency)
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
        
#         # QKV projection
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
        
#         # Attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)
        
#         # Apply attention to values
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
        
#         return x
    
#     def load_from_state_dict(self, state_dict: dict, prefix: str):
#         """Load attention weights from state_dict."""
#         if f'{prefix}qkv.weight' in state_dict:
#             self.qkv.weight.data = state_dict[f'{prefix}qkv.weight']
#             self.qkv.bias.data = state_dict[f'{prefix}qkv.bias']
#         if f'{prefix}proj.weight' in state_dict:
#             self.proj.weight.data = state_dict[f'{prefix}proj.weight']
#             self.proj.bias.data = state_dict[f'{prefix}proj.bias']




