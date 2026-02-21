"""
Simple linear segmentation head – a minimal baseline for comparison.

Registered as ``"linear_head"`` in the model registry.
"""

from torch import nn
from offroad_training_pipeline.models.registry import register_model


@register_model("linear_head")
class LinearHead(nn.Module):
    """Single 1×1 conv (equivalent to a per-token linear layer).

    Useful as the simplest possible baseline.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of each patch token.
    out_channels : int
        Number of segmentation classes.
    token_w, token_h : int
        Spatial patch grid dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_w: int,
        token_h: int,
    ):
        super().__init__()
        self.H = token_h
        self.W = token_w
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.head(x)
