"""
SegFormer-style transformer segmentation head.

Uses multi-head self-attention to capture global context, combined with
lightweight MLP decoders — the core idea from Xie et al. "SegFormer"
(NeurIPS 2021), adapted to work on DINOv2 patch tokens.

Registered as ``"segformer_head"`` in the model registry.
"""

import torch
from torch import nn
import torch.nn.functional as F
from offroad_training_pipeline.models.registry import register_model


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with spatial reduction (like SegFormer).

    Reduces the spatial resolution of K and V by ``sr_ratio`` before
    computing attention, making it efficient for dense prediction.
    """

    def __init__(self, dim, num_heads=8, sr_ratio=2, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial reduction for K, V
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """x: (B, N, C) where N = H * W."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_sr).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.sr_norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    """Mix-FFN from SegFormer: FC → 3×3 DWConv → GELU → FC.

    The depthwise conv injects local spatial information into the
    feed-forward network, compensating for the lack of positional encoding.
    """

    def __init__(self, dim, expansion=4, drop=0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single SegFormer encoder block: Attention + Mix-FFN with pre-norm."""

    def __init__(self, dim, num_heads=8, sr_ratio=2, expansion=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio,
            attn_drop=drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion=expansion, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

@register_model("segformer_head")
class SegFormerHead(nn.Module):
    """SegFormer-style transformer segmentation head.

    Architecture
    ------------
    1. **Project** backbone tokens to ``hidden_dim``
    2. **Transformer encoder** — ``depth`` blocks of efficient self-attention
       + Mix-FFN (captures global context across the entire image)
    3. **MLP decoder** — fuse features → class logits

    Why this works for off-road perception
    ---------------------------------------
    - Self-attention lets every patch see every other patch → understands
      that "sky is above landscape" and "rocks sit on ground"
    - Mix-FFN adds local spatial info without positional embeddings
    - Spatial-reduction attention keeps it efficient at dense resolution

    Parameters
    ----------
    in_channels : int
        Embedding dim from backbone (e.g. 384 for ViT-S/14).
    out_channels : int
        Number of segmentation classes.
    token_w, token_h : int
        Spatial patch grid dimensions.
    hidden_dim : int
        Internal feature width (default 256).
    depth : int
        Number of transformer blocks (default 4).
    num_heads : int
        Number of attention heads (default 8).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w

        # 1. Project backbone dim → hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 2. Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                sr_ratio=2 if i < depth // 2 else 1,  # reduce early, full attn later
                expansion=4,
                drop=0.1,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. MLP decoder → class logits
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape

        # Project
        x = self.proj(x)  # (B, N, hidden_dim)

        # Transformer blocks (operate in sequence space)
        for block in self.blocks:
            x = block(x, self.H, self.W)
        x = self.norm(x)

        # Reshape to spatial and decode
        x = x.permute(0, 2, 1).reshape(B, -1, self.H, self.W)
        return self.decoder(x)
