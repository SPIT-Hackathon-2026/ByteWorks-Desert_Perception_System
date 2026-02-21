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
NUM_CLASSES = 4

# Class definitions
CLASS_NAMES = ["Driveable", "Vegetation", "Obstacle", "Sky"]

# RGB colour palette
COLOR_PALETTE = np.array([
    [0, 200, 0],      # Driveable  – green
    [200, 0, 200],    # Vegetation – magenta/purple
    [255, 50, 50],    # Obstacle   – red
    [135, 206, 235],  # Sky        – sky blue
], dtype=np.uint8)

# Input normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
