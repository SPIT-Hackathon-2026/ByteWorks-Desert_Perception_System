"""
MLP segmentation head — two hidden layers with GELU activation.

A step up from linear_head while staying lightweight.
Registered as ``"mlp_head"`` in the model registry.
"""

from torch import nn
from offroad_training_pipeline.models.registry import register_model


@register_model("mlp_head")
class MLPHead(nn.Module):
    """Per-token MLP projected through 1×1 convolutions.

    Parameters
    ----------
    in_channels : int
        Patch-token embedding dim.
    out_channels : int
        Number of segmentation classes.
    token_w, token_h : int
        Spatial patch grid size.
    hidden_dim : int
        Width of hidden layers (default 256).
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
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.head(x)
