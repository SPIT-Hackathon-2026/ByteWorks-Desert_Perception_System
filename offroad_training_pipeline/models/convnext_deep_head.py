"""
Deeper ConvNeXt segmentation head with two residual-style blocks and
BatchNorm → more capacity than the single-block convnext_head.

Registered as ``"convnext_deep_head"`` in the model registry.
"""

from torch import nn
from offroad_training_pipeline.models.registry import register_model


class _ConvNeXtBlock(nn.Module):
    """Depthwise-separable conv block with residual connection."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


@register_model("convnext_deep_head")
class ConvNeXtDeepHead(nn.Module):
    """Two-block ConvNeXt head with residual connections and batch norm.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    token_w, token_h : int
    hidden_dim : int  (default 192)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
        hidden_dim: int = 192,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            _ConvNeXtBlock(hidden_dim),
            _ConvNeXtBlock(hidden_dim),
        )

        self.classifier = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)
