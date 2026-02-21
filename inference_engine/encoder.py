"""
ConvNeXt encoder wrapper for U-MixFormer.
"""

import torch
import torch.nn as nn
import timm

class ConvNeXtEncoder(nn.Module):
    def __init__(self, model_name: str = "convnext_tiny.fb_in22k", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)
