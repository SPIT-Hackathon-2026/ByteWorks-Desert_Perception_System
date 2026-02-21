"""
Central configuration for the offroad segmentation pipeline.

All hyperparameters, paths, class definitions, and color palettes live here.
Import this module everywhere to keep settings consistent.
"""

import os
import torch
import numpy as np

# ============================================================================
# Paths  (resolved relative to the *repo* root, not this file)
# ============================================================================
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_ROOT = os.path.join(_REPO_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "val")
TEST_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_testImages")

OUTPUT_DIR = os.path.join(_REPO_ROOT, "offroad_training_pipeline", "train_stats")
PREDICTIONS_DIR = os.path.join(_REPO_ROOT, "offroad_training_pipeline", "predictions")
MODEL_SAVE_DIR = os.path.join(_REPO_ROOT, "offroad_training_pipeline", "checkpoints")

# ============================================================================
# Device
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Image dimensions  (must be divisible by the patch size, i.e. 14 for DINOv2)
# ============================================================================
PATCH_SIZE = 14
ORIGINAL_W = 960
ORIGINAL_H = 540
IMG_W = int(((ORIGINAL_W / 2) // PATCH_SIZE) * PATCH_SIZE)  # 476
IMG_H = int(((ORIGINAL_H / 2) // PATCH_SIZE) * PATCH_SIZE)  # 266

# ============================================================================
# Training hyper-parameters
# ============================================================================
BATCH_SIZE = 2
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 8
NUM_WORKERS = 0  # increase on machines with more cores

# ============================================================================
# Backbone settings
# ============================================================================
BACKBONE_SIZE = "small"  # one of: small | base | large | giant
BACKBONE_ARCHS = {
    "small": "vits14",
    "base": "vitb14_reg",
    "large": "vitl14_reg",
    "giant": "vitg14_reg",
}
BACKBONE_REPO = "facebookresearch/dinov2"

# ============================================================================
# Class definitions  —  3 super-classes for UGV navigation
# ============================================================================
# Step 1: raw mask pixel value  →  original 10-class id  (for loading PNGs)
VALUE_MAP = {
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    600: 5,      # Flowers
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape (general ground)
    10000: 9,    # Sky
}

# Step 2: original 10-class id  →  5 super-class id
#   0 = Driveable    (Dry Grass, Ground Clutter, Landscape)
#   1 = Vegetation   (Trees, Lush Bushes, Dry Bushes, Flowers)
#   2 = Obstacle     (Logs, Ground Clutter items)
#   3 = Rocks        (Rocks — separated for finer obstacle understanding)
#   4 = Sky
CLASS_REMAP = {
    0: 1,   # Trees       → Vegetation
    1: 1,   # Lush Bushes → Vegetation
    2: 0,   # Dry Grass   → Driveable
    3: 1,   # Dry Bushes  → Vegetation
    4: 2,   # Ground Clutter → Obstacle
    5: 1,   # Flowers     → Vegetation
    6: 2,   # Logs        → Obstacle
    7: 3,   # Rocks       → Rocks
    8: 0,   # Landscape   → Driveable
    9: 4,   # Sky         → Sky
}

NUM_CLASSES = 5

CLASS_NAMES = [
    "Driveable",    # 0 — ground the UGV can traverse
    "Vegetation",   # 1 — bushes, trees, flowers (soft obstacles)
    "Obstacle",     # 2 — logs, clutter (hard obstacles, must avoid)
    "Rocks",        # 3 — rocks (hard obstacle, distinct from logs)
    "Sky",          # 4 — above the horizon
]

# RGB colour palette for visualisation (one per super-class)
COLOR_PALETTE = np.array(
    [
        [0, 200, 0],      # Driveable  – green
        [255, 165, 0],    # Vegetation – orange
        [255, 255, 0],    # Obstacle   – yellow
        [220, 30, 30],    # Rocks      – red
        [135, 206, 235],  # Sky        – sky blue
    ],
    dtype=np.uint8,
)

# ============================================================================
# ImageNet normalisation stats (used by DINOv2)
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
