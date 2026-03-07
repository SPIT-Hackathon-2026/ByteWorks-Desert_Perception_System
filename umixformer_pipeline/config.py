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
NUM_EPOCHS = 20
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
# Class definitions — full 10-class desert semantics from the hackathon
# ============================================================================
# Raw mask pixel value → dense 0–9 class id
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

# For training/inference we keep all 10 classes separate (identity remap)
CLASS_REMAP = {i: i for i in range(10)}

NUM_CLASSES = 10

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

# RGB colour palette for visualisation (matches test_segmentation.py)
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

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
