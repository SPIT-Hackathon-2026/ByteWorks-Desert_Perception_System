"""
Hybrid multi-scale segmentation head with ASPP + channel/spatial attention.

Combines Atrous Spatial Pyramid Pooling (multi-scale context), Squeeze-and-
Excitation channel attention, spatial attention gating, and a progressive
decoder.  Designed for dense pixel-level terrain classification AND object
identification in off-road scenes.

Registered as ``"hybrid_head"`` in the model registry.
"""

import torch
from torch import nn
import torch.nn.functional as F
from offroad_training_pipeline.models.registry import register_model


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBNGELU(nn.Module):
    """Conv2d → BatchNorm → GELU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures context at multiple scales.

    Four parallel branches with dilation rates 1, 3, 6, 12 plus a global
    average pooling branch, all fused into a single representation.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4

        self.branch1 = ConvBNGELU(in_ch, mid, kernel_size=1, dilation=1)
        self.branch2 = ConvBNGELU(in_ch, mid, kernel_size=3, dilation=3)
        self.branch3 = ConvBNGELU(in_ch, mid, kernel_size=3, dilation=6)
        self.branch4 = ConvBNGELU(in_ch, mid, kernel_size=3, dilation=12)

        # Global context via adaptive pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GELU(),
        )

        # Fuse all 5 branches → out_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        gp = F.interpolate(self.global_pool(x), size=(h, w),
                           mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([b1, b2, b3, b4, gp], dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    """Lightweight spatial attention gate (mean + max channel pooling)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True).values
        gate = self.conv(torch.cat([avg_pool, max_pool], dim=1))
        return x * gate


class ResidualBlock(nn.Module):
    """Conv → BN → GELU → Conv → BN + skip connection."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad,
                      groups=channels, bias=False),  # depthwise
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),  # pointwise
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

@register_model("hybrid_head")
class HybridSegHead(nn.Module):
    """Multi-scale hybrid segmentation head for off-road terrain perception.

    Architecture
    ------------
    1. **Projection** — reduce backbone dim to ``hidden_dim``
    2. **ASPP** — multi-scale context (dilations 1, 3, 6, 12 + global pool)
    3. **Channel attention** — SE block to focus on relevant feature channels
    4. **Spatial attention** — highlight important spatial regions
    5. **Residual decoder** — two residual blocks for refinement
    6. **Classifier** — 1×1 conv to class logits

    This gives the model the ability to:
    • Capture fine-grained detail (small dilation / 1×1 conv)
    • Understand large-scale context (large dilation / global pool)
    • Focus on the most relevant channels and spatial regions
    • Identify and localise objects of different sizes

    Parameters
    ----------
    in_channels : int
        Embedding dim of each patch token (e.g. 384 for ViT-S/14).
    out_channels : int
        Number of segmentation classes.
    token_w, token_h : int
        Spatial dimensions of the patch-token grid.
    hidden_dim : int
        Internal feature width (default 256).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w

        # 1. Project backbone features to hidden_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # 2. Multi-scale context
        self.aspp = ASPPModule(hidden_dim, hidden_dim)

        # 3. Channel attention
        self.channel_attn = SEBlock(hidden_dim)

        # 4. Spatial attention
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # 5. Residual decoder
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim, kernel_size=7),
            ResidualBlock(hidden_dim, kernel_size=5),
        )

        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim // 2, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        x = self.proj(x)
        x = self.aspp(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.decoder(x)
        return self.classifier(x)
