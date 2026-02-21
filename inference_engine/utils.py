"""
Mask conversion and colour-mapping utilities for inference.
"""

import numpy as np
import torch
from inference_engine.config import COLOR_PALETTE, NUM_CLASSES

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask (H, W) -> RGB colour image (H, W, 3)."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = COLOR_PALETTE[c]
    return color

def denormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Un-normalise a (C, H, W) tensor back to [0, 255] uint8 numpy."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return img
