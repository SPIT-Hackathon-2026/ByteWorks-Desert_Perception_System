"""
Mask conversion and colour-mapping utilities.
"""

import numpy as np
from PIL import Image

from umixformer_pipeline.config import VALUE_MAP, CLASS_REMAP, COLOR_PALETTE, NUM_CLASSES


def convert_mask(mask_pil: Image.Image) -> Image.Image:
    """Convert a raw segmentation PNG to 4-class super-class labels.

    Pipeline:
        1. Read 16-bit pixel values
        2. Map via VALUE_MAP → 10-class IDs
        3. Remap via CLASS_REMAP → 4 super-class IDs
    """
    mask_np = np.array(mask_pil, dtype=np.int32)

    # Step 1: raw pixel values → 10-class IDs
    class_mask = np.zeros_like(mask_np, dtype=np.uint8)
    for pixel_val, class_id in VALUE_MAP.items():
        class_mask[mask_np == pixel_val] = class_id

    # Step 2: 10-class → 4 super-class
    super_mask = np.zeros_like(class_mask, dtype=np.uint8)
    for old_id, new_id in CLASS_REMAP.items():
        super_mask[class_mask == old_id] = new_id

    return Image.fromarray(super_mask, mode="L")


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask (H, W) → RGB colour image (H, W, 3)."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = COLOR_PALETTE[c]
    return color


def denormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Un-normalise a (C, H, W) tensor back to [0, 255] uint8 numpy."""
    import torch
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return img
