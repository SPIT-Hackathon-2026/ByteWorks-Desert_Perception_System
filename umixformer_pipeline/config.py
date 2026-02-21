"""
Central configuration for the U-MixFormer off-road segmentation pipeline.

All hyperparameters, paths, class definitions, and color palettes.
"""

import os
import torch
import numpy as np

# ============================================================================
# Paths
# ============================================================================
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_ROOT = os.path.join(_REPO_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "val")
TEST_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_testImages")

OUTPUT_DIR = os.path.join(_REPO_ROOT, "umixformer_pipeline", "train_stats")
PREDICTIONS_DIR = os.path.join(_REPO_ROOT, "umixformer_pipeline", "predictions")
MODEL_SAVE_DIR = os.path.join(_REPO_ROOT, "umixformer_pipeline", "checkpoints")

# ============================================================================
# Device
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Image dimensions — 384×384 (balances detail vs. VRAM on 6 GB GPU)
# ============================================================================
IMG_SIZE = 384  # square crop/resize

# ============================================================================
# Training hyper-parameters
# ============================================================================
BATCH_SIZE = 2
LEARNING_RATE = 6e-5          # lower LR for fine-tuning pretrained encoder
ENCODER_LR_MULT = 0.1        # encoder trains at 10× lower LR
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 4          # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 8

# ============================================================================
# Encoder settings (ConvNeXt-Tiny from timm → 4-stage hierarchical features)
# ============================================================================
ENCODER_NAME = "convnext_tiny.fb_in22k"
ENCODER_CHANNELS = [96, 192, 384, 768]   # ConvNeXt-Tiny stage dims
DECODER_DIM = 128                        # unified decoder channel dim (128 for 6GB GPU)

# ============================================================================
# U-MixFormer decoder settings
# ============================================================================
NUM_HEADS = 4
MIX_ATTN_DROP = 0.0
FFN_EXPANSION = 4
FFN_DROP = 0.1
DECODER_DEPTH = 1             # transformer blocks per decoder stage

# ============================================================================
# Loss settings
# ============================================================================
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_SMOOTH = 1.0
LOSS_WEIGHTS = {"focal": 1.0, "dice": 1.0}  # combined loss weighting

# ============================================================================
# Class definitions — 4 super-classes for UGV navigation
# ============================================================================
# Raw mask pixel value → original 10-class id
VALUE_MAP = {
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    600: 5,      # Flowers
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9,    # Sky
}

# 10-class → 3 super-class mapping
#   0 = Terrain   (Dry Grass, Landscape — open ground the UGV can traverse)
#   1 = Obstacle  (Trees, Bushes, Flowers, Ground Clutter, Logs, Rocks —
#                  anything the UGV cannot drive through)
#   2 = Sky
CLASS_REMAP = {
    0: 1,   # Trees         → Obstacle
    1: 1,   # Lush Bushes   → Obstacle
    2: 1,   # Dry Grass     → Terrain
    3: 1,   # Dry Bushes    → Obstacle
    4: 1,   # Ground Clutter→ Obstacle
    5: 1,   # Flowers       → Obstacle
    6: 1,   # Logs          → Obstacle
    7: 1,   # Rocks         → Obstacle
    8: 0,   # Landscape     → Terrain
    9: 2,   # Sky           → Sky
}

NUM_CLASSES = 3

CLASS_NAMES = [
    "Terrain",      # 0
    "Obstacle",     # 1
    "Sky",          # 2
]

# RGB colour palette for visualisation
COLOR_PALETTE = np.array([
    [0, 200, 0],      # Terrain   – green
    [220, 50, 50],    # Obstacle  – red
    [135, 206, 235],  # Sky       – sky blue
], dtype=np.uint8)

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
