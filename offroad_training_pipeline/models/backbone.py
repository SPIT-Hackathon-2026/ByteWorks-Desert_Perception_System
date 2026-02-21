"""
DINOv2 backbone loader.

The backbone is always frozen – only the segmentation head is trained.
"""

import torch
from offroad_training_pipeline.config import BACKBONE_ARCHS, BACKBONE_REPO


def load_backbone(size: str = "small", device: torch.device | None = None):
    """Load a frozen DINOv2 ViT backbone.

    Parameters
    ----------
    size : str
        One of ``small``, ``base``, ``large``, ``giant``.
    device : torch.device, optional
        Target device (defaults to CPU).

    Returns
    -------
    backbone : nn.Module
        The frozen DINOv2 model ready for feature extraction.
    """
    if size not in BACKBONE_ARCHS:
        raise ValueError(f"Unknown backbone size '{size}'. Choose from {list(BACKBONE_ARCHS)}")

    arch = BACKBONE_ARCHS[size]
    model_name = f"dinov2_{arch}"

    print(f"Loading DINOv2 backbone ({model_name})…")
    backbone = torch.hub.load(repo_or_dir=BACKBONE_REPO, model=model_name)
    backbone.eval()

    if device is not None:
        backbone = backbone.to(device)

    # Freeze all parameters
    for p in backbone.parameters():
        p.requires_grad = False

    print("Backbone loaded & frozen ✓")
    return backbone


def get_embedding_dim(backbone, sample_tensor: torch.Tensor) -> int:
    """Run a single forward pass to discover the patch-token embedding dim."""
    device = next(backbone.parameters()).device
    with torch.no_grad():
        tokens = backbone.forward_features(sample_tensor.to(device))["x_norm_patchtokens"]
    return tokens.shape[2]
