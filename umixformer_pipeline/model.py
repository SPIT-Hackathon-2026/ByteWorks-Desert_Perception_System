"""
Full U-MixFormer segmentation model.

Combines:
  - ConvNeXt encoder (4-stage hierarchical features)
  - U-MixFormer decoder (mix-attention + U-Net structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from umixformer_pipeline.encoder import ConvNeXtEncoder
from umixformer_pipeline.decoder import UMixFormerDecoder
from umixformer_pipeline.config import (
    ENCODER_NAME,
    ENCODER_CHANNELS,
    DECODER_DIM,
    NUM_HEADS,
    FFN_EXPANSION,
    MIX_ATTN_DROP,
    FFN_DROP,
    DECODER_DEPTH,
    NUM_CLASSES,
)


class UMixFormerSeg(nn.Module):
    """Complete U-MixFormer segmentation model.

    Parameters
    ----------
    encoder_name : str
        ConvNeXt model name from timm.
    pretrained_encoder : bool
        Use ImageNet-pretrained encoder weights.
    decoder_dim : int
        Unified channel dimension inside the decoder.
    num_heads : int
        Number of attention heads in mix-attention.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        pretrained_encoder: bool = True,
        decoder_dim: int = DECODER_DIM,
        num_heads: int = NUM_HEADS,
        ffn_expansion: int = FFN_EXPANSION,
        attn_drop: float = MIX_ATTN_DROP,
        ffn_drop: float = FFN_DROP,
        decoder_depth: int = DECODER_DEPTH,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(encoder_name, pretrained=pretrained_encoder)
        encoder_channels = self.encoder.channels

        self.decoder = UMixFormerDecoder(
            encoder_channels=encoder_channels,
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
            decoder_depth=decoder_depth,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) — input images

        Returns
        -------
        logits : (B, num_classes, H, W) — upsampled to input resolution
        """
        H, W = x.shape[2], x.shape[3]

        # Encoder: multi-scale features
        enc_features = self.encoder(x)

        # Decoder: mix-attention U-Net
        logits = self.decoder(enc_features)

        # Upsample to full resolution
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits

    def get_param_groups(self, encoder_lr_mult: float = 0.1):
        """Return parameter groups with differential learning rates.

        Encoder gets lower LR since it's pretrained.
        Decoder gets full LR since it's trained from scratch.
        """
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())

        return [
            {"params": encoder_params, "lr_mult": encoder_lr_mult},
            {"params": decoder_params, "lr_mult": 1.0},
        ]
