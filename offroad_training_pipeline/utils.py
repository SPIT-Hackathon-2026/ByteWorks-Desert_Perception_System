"""
Shared utility helpers for image I/O and mask manipulation.
"""

import cv2
import numpy as np
from PIL import Image

from offroad_training_pipeline.config import (
    CLASS_REMAP,
    COLOR_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    VALUE_MAP,
)


# ============================================================================
# Image helpers
# ============================================================================

def save_image(img_tensor, filename: str) -> None:
    """Save a CHW image tensor (ImageNet-normalised) to *filename* after
    denormalising back to uint8 BGR for OpenCV."""
    img = np.array(img_tensor)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = np.moveaxis(img, 0, -1)           # CHW → HWC
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])   # RGB → BGR


def denormalize_image(img_tensor) -> np.ndarray:
    """Return a float32 HWC [0, 1] numpy array from a CHW normalised tensor."""
    img = img_tensor.cpu().numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


# ============================================================================
# Mask helpers
# ============================================================================

def convert_mask(mask: Image.Image) -> Image.Image:
    """Map raw pixel values → 10-class IDs (VALUE_MAP) → 3 super-class IDs (CLASS_REMAP)."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, class_id in VALUE_MAP.items():
        super_id = CLASS_REMAP[class_id]
        new_arr[arr == raw_value] = super_id
    return Image.fromarray(new_arr)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a HxW class-id mask to an HxWx3 RGB colour image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(NUM_CLASSES):
        color[mask == cid] = COLOR_PALETTE[cid]
    return color
