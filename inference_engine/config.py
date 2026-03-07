"""
Configuration for the U-MixFormer inference engine.
"""

import torch
import numpy as np

# Image dimensions
IMG_SIZE = 384

# Encoder / Decoder settings
ENCODER_NAME = "convnext_tiny.fb_in22k"
ENCODER_CHANNELS = [96, 192, 384, 768]
DECODER_DIM = 128
NUM_HEADS = 4
DECODER_DEPTH = 1

# We train and infer with the full 10 hackathon classes
# 0: Trees, 1: Lush Bushes, 2: Dry Grass, 3: Dry Bushes,
# 4: Ground Clutter, 5: Flowers, 6: Logs, 7: Rocks,
# 8: Landscape, 9: Sky
NUM_CLASSES = 10

# Class definitions (ordered to match VALUE_MAP / training pipeline)
CLASS_NAMES = [
    "Trees",          # 0
    "Lush Bushes",    # 1
    "Dry Grass",      # 2
    "Dry Bushes",     # 3
    "Ground Clutter", # 4
    "Flowers",        # 5
    "Logs",           # 6
    "Rocks",          # 7
    "Landscape",      # 8
    "Sky",            # 9
]

# RGB colour palette aligned with the hackathon documentation
COLOR_PALETTE = np.array([
    [34, 139, 34],   # Trees - forest green
    [0, 255, 0],     # Lush Bushes - lime
    [210, 180, 140], # Dry Grass - tan
    [139, 90, 43],   # Dry Bushes - brown
    [128, 128, 0],   # Ground Clutter - olive
    [255, 192, 203], # Flowers - pink
    [139, 69, 19],   # Logs - saddle brown
    [128, 128, 128], # Rocks - gray
    [160, 82, 45],   # Landscape - sienna
    [135, 206, 235], # Sky - sky blue
], dtype=np.uint8)

# Input normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
