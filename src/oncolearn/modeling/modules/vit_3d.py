"""3D Vision Transformer for loading FM-BCMRI pretrained checkpoints.

Vendors the minimal architecture code from the FM-BCMRI submodule
(submodules/FM-BCMRI/fmbcmri/lib/) so the checkpoint can be loaded with
only `timm` as an external dependency — no sys.path manipulation required.

Source: https://github.com/zhenweishi/FM-BCMRI
License: CC BY-NC 4.0 (non-commercial)
"""
from __future__ import annotations

import math
from functools import partial, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from timm.layers import to_3tuple
from timm.models.vision_transformer import VisionTransformer

__all__ = ["PatchEmbed3D", "VisionTransformerMoCo3D", "vit_3d_base_patchsize8"]


class PatchEmbed3D(nn.Module):
    """3D image to patch embedding via Conv3d.

    Expects input shape ``(B, C, H, W, D)``.
    Adapted from ``fmbcmri/lib/layers/patch_embed.py``.
    """

    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
        bias: bool = True,
        **kwargs,  # absorb extra timm kwargs (dynamic_img_pad, etc.)
    ):
        super().__init__()
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]
        self.num_patches = int(np.prod(self.grid_size))
        self.flatten = flatten
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, h, w, d)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return self.norm(x)


def _build_3d_sincos_pos_embed(
    grid_size,
    embed_dim: int,
    num_tokens: int = 1,
    temperature: float = 10000.0,
) -> nn.Parameter:
    """Fixed 3D sin-cos positional embedding.

    Adapted from ``fmbcmri/lib/layers/position_embed.py``.
    """
    h, w, d = to_3tuple(grid_size)
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3D sin-cos position embedding"
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    grid_h, grid_w, grid_d = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        torch.arange(d, dtype=torch.float32),
        indexing="ij",
    )
    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_d = torch.einsum("m,d->md", grid_d.flatten(), omega)
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h),
         torch.sin(out_w), torch.cos(out_w),
         torch.sin(out_d), torch.cos(out_d)],
        dim=1,
    )[None]  # (1, num_patches, embed_dim)
    if num_tokens == 1:
        pe_token = torch.zeros(1, 1, embed_dim)
        pos_emb = torch.cat([pe_token, pos_emb], dim=1)
    param = nn.Parameter(pos_emb)
    param.requires_grad = False
    return param


class VisionTransformerMoCo3D(VisionTransformer):
    """timm VisionTransformer adapted for 3D MRI volumes.

    Replaces the standard 2D positional embedding with a fixed 3D sin-cos
    variant and uses ``PatchEmbed3D`` for patch tokenisation.

    Adapted from ``fmbcmri/lib/models/vision_transformer_moco.py``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_embed = _build_3d_sincos_pos_embed(
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
            num_tokens=self.num_prefix_tokens,
        )
        # MoCo-style xavier init for patch embedding
        val = math.sqrt(
            6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
        )
        nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
        nn.init.zeros_(self.patch_embed.proj.bias)


def vit_3d_base_patchsize8(**kwargs) -> VisionTransformerMoCo3D:
    """ViT-Base/8 for 3D volumes — matches the FM-BCMRI pretrained checkpoint."""
    return VisionTransformerMoCo3D(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
