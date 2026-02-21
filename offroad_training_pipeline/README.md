# Off-Road Segmentation Pipeline

A modular, extensible PyTorch segmentation pipeline built on DINOv2 features for off-road terrain classification.

## Directory Structure

```
SPIT_Hackathon/
├── dataset/
│   ├── Offroad_Segmentation_Training_Dataset/
│   │   ├── train/
│   │   │   ├── Color_Images/
│   │   │   └── Segmentation/
│   │   └── val/
│   │       ├── Color_Images/
│   │       └── Segmentation/
│   └── Offroad_Segmentation_testImages/
│       ├── Color_Images/
│       └── Segmentation/
│
├── offroad_training_pipeline/       # ← Refactored modular code
│   ├── __init__.py
│   ├── config.py                    # Central configuration
│   ├── dataset.py                   # Data loading & transforms
│   ├── utils.py                     # Image/mask utilities
│   ├── metrics.py                   # IoU, Dice, Pixel Accuracy
│   ├── visualization.py             # Plots & comparison images
│   ├── train.py                     # Training script (CLI)
│   ├── test.py                      # Inference script (CLI)
│   ├── visualize.py                 # Mask colorisation script (CLI)
│   ├── checkpoints/                 # Saved model weights
│   ├── train_stats/                 # Training history & plots
│   ├── predictions/                 # Inference outputs
│   └── models/
│       ├── __init__.py
│       ├── registry.py              # Model registry system
│       ├── backbone.py              # DINOv2 loader
│       ├── convnext_head.py         # ConvNeXt-style head (default)
│       └── linear_head.py           # Minimal 1×1 conv baseline
│
├── Offroad_Segmentation_Scripts/    # Original standalone scripts
│   ├── train_segmentation.py
│   ├── test_segmentation.py
│   └── visualize.py
│
├── main.py                          # Convenience CLI entry-point
└── README.md                        # This file
```

## Quick Start

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
# or: .venv\Scripts\activate  (on Windows)
```

### 2. Train a Segmentation Head

**Using the refactored pipeline:**

```bash
# Train with ConvNeXt head (default)
python main.py train --model convnext_head --epochs 10

# Train with linear baseline
python main.py train --model linear_head --epochs 20 --lr 1e-3

# All options
python -m offroad_training_pipeline.train --help
```

Outputs:
- Model checkpoint → `offroad_training_pipeline/checkpoints/convnext_head.pth`
- Training history → `offroad_training_pipeline/train_stats/evaluation_metrics.txt`
- Training plots → `offroad_training_pipeline/train_stats/*.png`

**Using the original script:**

```bash
cd Offroad_Segmentation_Scripts
python train_segmentation.py
```

This will save the model to `offroad_training_pipeline/checkpoints/segmentation_head.pth`.

### 3. Run Inference / Evaluation

**Using the refactored pipeline:**

```bash
python main.py test --model convnext_head \
  --model_path offroad_training_pipeline/checkpoints/convnext_head.pth

# Or with defaults
python -m offroad_training_pipeline.test
```

Outputs:
- Raw masks (class IDs) → `offroad_training_pipeline/predictions/masks/`
- Coloured masks (RGB) → `offroad_training_pipeline/predictions/masks_color/`
- Comparison images → `offroad_training_pipeline/predictions/comparisons/`
- Metrics → `offroad_training_pipeline/predictions/evaluation_metrics.txt`

**Using the original script:**

```bash
cd Offroad_Segmentation_Scripts
python test_segmentation.py --model_path ../offroad_training_pipeline/checkpoints/convnext_head.pth
```

### 4. Visualise Masks

```bash
python main.py visualize
# or
python -m offroad_training_pipeline.visualize
```

## Configuration

All paths, hyperparameters, and class definitions are centralised in:

```python
# offroad_training_pipeline/config.py
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_CLASSES = 10
IMG_H, IMG_W = 266, 476
```

To change defaults, either:
1. Edit `offroad_training_pipeline/config.py`, or
2. Pass command-line flags (which override config defaults)

## Class Definitions

| ID | Class | Colour |
|---|---|---|
| 0 | Background | Black |
| 1 | Trees | Forest Green |
| 2 | Lush Bushes | Lime |
| 3 | Dry Grass | Tan |
| 4 | Dry Bushes | Brown |
| 5 | Ground Clutter | Olive |
| 6 | Logs | Saddle Brown |
| 7 | Rocks | Gray |
| 8 | Landscape | Sienna |
| 9 | Sky | Sky Blue |

## Model Architecture

### Backbone
- **DINOv2 ViT-S/14** (frozen)
- Extracts patch-token features from images
- Available sizes: `small` (default), `base`, `large`, `giant`

### Segmentation Heads

#### ConvNeXt Head (default)
```
Patch Tokens (B, N, 384)
    ↓ reshape to spatial grid
(B, C, H, W)
    ↓ 7×7 conv + GELU
(B, 128, H, W)
    ↓ depthwise 7×7 + 1×1 convs + GELU
(B, 128, H, W)
    ↓ 1×1 classifier
(B, 10, H, W)
    ↓ bilinear interpolate to input size
```

#### Linear Head (baseline)
```
Patch Tokens (B, N, 384)
    ↓ reshape to spatial grid
(B, C, H, W)
    ↓ 1×1 classifier
(B, 10, H, W)
    ↓ bilinear interpolate to input size
```

## Adding a New Segmentation Head

1. Create a new file in `offroad_training_pipeline/models/` (e.g., `my_head.py`):

```python
from torch import nn
from offroad_training_pipeline.models.registry import register_model

@register_model("my_head")
class MyHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_w, token_h):
        super().__init__()
        self.H, self.W = token_h, token_w
        # Your architecture here
        self.classifier = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        # Your layers here
        return self.classifier(x)
```

2. Import it in `offroad_training_pipeline/models/__init__.py`:

```python
import offroad_training_pipeline.models.my_head  # noqa: F401
```

3. Use it:

```bash
python main.py train --model my_head --epochs 10
```

## Metrics

All metrics are computed per-pixel across the entire dataset:

- **Intersection-over-Union (IoU)** – Per-class overlap ratio
- **Dice Score (F1)** – Harmonic mean of precision & recall
- **Pixel Accuracy** – Fraction of correctly classified pixels

Per-class and mean values are reported in `evaluation_metrics.txt`.

## Training

### Loss Function
Cross-Entropy Loss (standard for multi-class segmentation)

### Optimizer
SGD with momentum (default: `lr=1e-4`, `momentum=0.9`)

### Data Augmentation
- Resize to 266×476 (preserves 14×34 patch grid)
- ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Performance Monitoring

During training, the console will show:

```
Training: 50%|████▌     | 5/10 [02:45<02:45, 33.00s/epoch]
Epoch 5/10 [Train]: 100%|██████████| 4/4 [00:33<00:00, 8.33s/batch]
Epoch 5/10 [Val]: 100%|██████████| 1/1 [00:08<00:00, 8.00s/batch]
```

Plots are saved to `train_stats/`:
- `training_curves.png` – Loss & Pixel Accuracy
- `iou_curves.png` – Per-epoch IoU trends
- `dice_curves.png` – Per-epoch Dice trends
- `all_metrics_curves.png` – Combined 2×2 overview

## Inference & Evaluation

Run `test.py` to:
1. Load a trained head
2. Forward each image through DINOv2 + head
3. Save predictions (raw + coloured)
4. Compute metrics
5. Generate comparison images
6. Plot per-class IoU chart

Output structure:

```
predictions/
├── masks/              # Raw (B&W) class-ID PNGs
├── masks_color/        # Coloured RGB PNGs
├── comparisons/        # 3-panel: input | truth | pred
├── evaluation_metrics.txt
└── per_class_metrics.png
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'offroad_training_pipeline'"
- Ensure you're running from the repo root: `cd /home/raj_99/Projects/SPIT_Hackathon`
- Or explicitly add to PYTHONPATH: `export PYTHONPATH="${PWD}:$PYTHONPATH"`

### Out of Memory (CUDA)
- Reduce `--batch_size` (default: 2)
- Use a smaller backbone: `--backbone_size small` (default)
- Enable mixed precision (not yet implemented)

### Paths not found
- All paths resolve from the repo root (`SPIT_Hackathon/`)
- Ensure `dataset/` subdirectories exist with correct structure (see tree above)

## References

- DINOv2: [Oquab et al., 2024](https://arxiv.org/abs/2304.07193)
- Vision Transformer: [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545)

## License

[Add your license here]

## Authors

[Add contributors]
