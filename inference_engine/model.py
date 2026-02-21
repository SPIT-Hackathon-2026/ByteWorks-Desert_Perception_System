"""
Complete U-MixFormer segmentation model for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference_engine.encoder import ConvNeXtEncoder
from inference_engine.decoder import UMixFormerDecoder
from inference_engine.config import (
    ENCODER_NAME,
    DECODER_DIM,
    NUM_HEADS,
    DECODER_DEPTH,
    NUM_CLASSES,
)

class UMixFormerSeg(nn.Module):
    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        pretrained_encoder: bool = False,
        decoder_dim: int = DECODER_DIM,
        num_heads: int = NUM_HEADS,
        decoder_depth: int = DECODER_DEPTH,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(encoder_name, pretrained=pretrained_encoder)
        self.decoder = UMixFormerDecoder(
            encoder_channels=self.encoder.channels,
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            decoder_depth=decoder_depth,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        enc_features = self.encoder(x)
        logits = self.decoder(enc_features)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
