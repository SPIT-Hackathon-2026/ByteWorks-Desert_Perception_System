"""
UPerNet-inspired multi-scale segmentation head.

Applies 1×1, 3×3, 5×5, and 7×7 convolutions in parallel, concatenates
their outputs, and fuses them — capturing both fine and coarse context.

Registered as ``"multiscale_head"`` in the model registry.
"""

import torch
from torch import nn
from offroad_training_pipeline.models.registry import register_model


class _ScaleBranch(nn.Module):
    """Single convolution branch at a specific kernel size."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


@register_model("multiscale_head")
class MultiScaleHead(nn.Module):
    """Multi-scale parallel-branch segmentation head.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    token_w, token_h : int
    branch_dim : int
        Output channels per branch (default 64).  Total fused width =
        ``4 * branch_dim``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
        branch_dim: int = 64,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w

        self.branches = nn.ModuleList([
            _ScaleBranch(in_channels, branch_dim, k) for k in [1, 3, 5, 7]
        ])

        fused_dim = branch_dim * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
            nn.Conv2d(fused_dim, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        return self.fuse(x)
