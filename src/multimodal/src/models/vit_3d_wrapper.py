# """
# Wrapper for 3D Vision Transformer from checkpoint.
# Adapts 3D ViT to work with 2D DICOM images by converting 2D slices to pseudo-3D volumes.
# """
# import logging
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# logger = logging.getLogger(__name__)


# class PatchEmbed3D(nn.Module):
#     """3D patch embedding layer, adapted for 2D input."""
#     def __init__(self, img_size=224, patch_size=8, in_chans=1, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.in_chans = in_chans
        
#         # 3D convolution for patch embedding
#         # Original: (768, 1, 8, 8, 8) - expects 3D input
#         # We'll adapt it to work with 2D by using 2D conv and stacking
#         self.proj_3d = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
#     def forward(self, x):
#         # x: (B, C, H, W) 2D input
#         B, C, H, W = x.shape
        
#         # Handle channel mismatch: if input has 3 channels but model expects 1, convert to grayscale
#         if C != self.in_chans:
#             if C == 3 and self.in_chans == 1:
#                 # Convert RGB to grayscale
#                 x = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
#             elif C == 1 and self.in_chans == 3:
#                 # Expand single channel to 3 channels
#                 x = x.repeat(1, 3, 1, 1)  # (B, 3, H, W)
#             else:
#                 # Take first channel or average
#                 x = x[:, 0:1, :, :] if C > self.in_chans else x
        
#         # Method: Convert 2D to pseudo-3D by repeating slices
#         # Stack multiple copies along depth to create pseudo-3D volume
#         depth = self.patch_size  # Use patch_size as depth
#         x_3d = x.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # (B, C, D, H, W)
        
#         # Apply 3D convolution
#         x = self.proj_3d(x_3d)  # (B, embed_dim, D', H', W')
#         B_out, E, D, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', E)
        
#         return x


# class ViT3DWrapper(nn.Module):
#     """
#     Wrapper for 3D Vision Transformer from checkpoint.
#     Adapts 3D ViT to work with 2D images by creating pseudo-3D volumes.
#     """
    
#     def __init__(self, state_dict: dict, freeze_backbone: bool = True):
#         super().__init__()
#         self.freeze_backbone = freeze_backbone
        
#         # Extract model parameters from state_dict
#         hidden_size = state_dict.get('cls_token', torch.zeros(1, 1, 768)).shape[-1]
        
#         # Count transformer blocks
#         block_keys = [k for k in state_dict.keys() if k.startswith('blocks.')]
#         num_layers = max([int(k.split('.')[1]) for k in block_keys if k.split('.')[1].isdigit()]) + 1
        
#         # Patch embedding (3D)
#         patch_embed_weight = state_dict.get('patch_embed.proj.weight')
#         if patch_embed_weight is not None:
#             patch_size = patch_embed_weight.shape[2]  # Assuming cubic patches (8, 8, 8)
#             in_chans = patch_embed_weight.shape[1]
#             embed_dim = patch_embed_weight.shape[0]
#         else:
#             patch_size = 8
#             in_chans = 1
#             embed_dim = hidden_size
        
#         self.patch_embed = PatchEmbed3D(
#             img_size=224,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim
#         )
        
#         # Load patch embedding weights directly (3D conv)
#         if 'patch_embed.proj.weight' in state_dict:
#             self.patch_embed.proj_3d.weight.data = state_dict['patch_embed.proj.weight']
#         if 'patch_embed.proj.bias' in state_dict:
#             self.patch_embed.proj_3d.bias.data = state_dict['patch_embed.proj.bias']
        
#         # CLS token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
#         if 'cls_token' in state_dict:
#             self.cls_token.data = state_dict['cls_token']
        
#         # Positional embedding
#         pos_embed = state_dict.get('pos_embed')
#         if pos_embed is not None:
#             # Original pos_embed shape: (1, 217, 768) for 3D
#             # We need to adapt it for 2D->3D conversion
#             # For now, use it as-is (may need interpolation for different patch counts)
#             num_patches = pos_embed.shape[1] - 1  # Exclude CLS token
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
#             self.pos_embed.data = pos_embed
#         else:
#             # Create default positional embedding
#             # For 2D->3D: (224/8)^2 * depth_patches
#             num_patches_2d = (224 // patch_size) ** 2
#             depth_patches = patch_size  # Depth dimension
#             num_patches = num_patches_2d * depth_patches
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
#             nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
#         # Transformer blocks
#         self.blocks = nn.ModuleList()
#         for i in range(num_layers):
#             block = self._create_transformer_block(state_dict, i, hidden_size)
#             self.blocks.append(block)
        
#         # Layer norm
#         self.norm = nn.LayerNorm(hidden_size)
#         if 'norm.weight' in state_dict:
#             self.norm.weight.data = state_dict['norm.weight']
#             self.norm.bias.data = state_dict['norm.bias']
        
#         # Freeze if requested
#         if freeze_backbone:
#             for param in self.parameters():
#                 param.requires_grad = False
    
#     def _create_transformer_block(self, state_dict: dict, layer_idx: int, hidden_size: int):
#         """Create a transformer block from state_dict."""
#         from .vit_block import TransformerBlock
        
#         # Infer num_heads from qkv weight shape
#         qkv_key = f'blocks.{layer_idx}.attn.qkv.weight'
#         if qkv_key in state_dict:
#             qkv_weight = state_dict[qkv_key]
#             # qkv weight: (3 * hidden_size, hidden_size) for combined QKV
#             num_heads = 12  # Default, or infer from shape
#             if qkv_weight.shape[0] == 3 * hidden_size:
#                 # Standard: 12 heads for 768 dim
#                 num_heads = 12
#         else:
#             num_heads = 12
        
#         block = TransformerBlock(hidden_size, num_heads=num_heads)
        
#         # Load weights if available
#         prefix = f'blocks.{layer_idx}.'
#         try:
#             block.load_from_state_dict(state_dict, prefix)
#         except Exception as e:
#             logger.warning(f"Could not load all weights for block {layer_idx}: {e}")
        
#         return block
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: (B, C, H, W) 2D images
        
#         Returns:
#             (B, hidden_size) CLS token embedding
#         """
#         B = x.shape[0]
        
#         # Patch embedding (handles 2D->3D conversion)
#         x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
#         # Add CLS token
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
#         x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, hidden_size)
        
#         # Add positional embedding (interpolate if needed)
#         if x.shape[1] != self.pos_embed.shape[1]:
#             # Interpolate positional embedding to match patch count
#             pos_embed = F.interpolate(
#                 self.pos_embed.transpose(1, 2),
#                 size=x.shape[1],
#                 mode='linear',
#                 align_corners=False
#             ).transpose(1, 2)
#         else:
#             pos_embed = self.pos_embed
        
#         x = x + pos_embed
        
#         # Apply transformer blocks
#         for block in self.blocks:
#             x = block(x)
        
#         # Layer norm
#         x = self.norm(x)
        
#         # Return CLS token (first token)
#         return x[:, 0]  # (B, hidden_size)

