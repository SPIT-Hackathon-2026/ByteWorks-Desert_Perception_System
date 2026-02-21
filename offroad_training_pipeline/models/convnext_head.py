"""
ConvNeXt-style segmentation head (the original architecture from the codebase).

Registered as ``"convnext_head"`` in the model registry.
"""

from torch import nn
from offroad_training_pipeline.models.registry import register_model


@register_model("convnext_head")
class ConvNeXtHead(nn.Module):
    """Lightweight ConvNeXt-inspired segmentation head.

    Accepts ``(B, N, C)`` patch-token features from a ViT backbone, reshapes
    them into a spatial grid, and produces per-pixel class logits.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of each patch token (e.g. 384 for ViT-S).
    out_channels : int
        Number of segmentation classes.
    token_w, token_h : int
        Width and height of the spatial patch grid.
    hidden_dim : int
        Width of the hidden convolutional layers (default 128).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)
