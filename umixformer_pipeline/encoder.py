"""
ConvNeXt encoder wrapper.

Extracts 4-stage hierarchical feature maps from a pretrained ConvNeXt backbone,
exactly the multi-resolution features required by U-MixFormer.

Stage outputs (for ConvNeXt-Tiny):
  Stage 1: (B, 96,  H/4,  W/4)
  Stage 2: (B, 192, H/8,  W/8)
  Stage 3: (B, 384, H/16, W/16)
  Stage 4: (B, 768, H/32, W/32)
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtEncoder(nn.Module):
    """Hierarchical 4-stage ConvNeXt encoder via timm.

    Parameters
    ----------
    model_name : str
        Any ConvNeXt variant from timm (e.g. "convnext_tiny.fb_in22k").
    pretrained : bool
        Load ImageNet-pretrained weights.
    """

    def __init__(self, model_name: str = "convnext_tiny.fb_in22k", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,      # extract intermediate features
            out_indices=(0, 1, 2, 3),
        )
        # Query the actual output channels
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of 4 feature maps at decreasing resolutions."""
        return self.backbone(x)
