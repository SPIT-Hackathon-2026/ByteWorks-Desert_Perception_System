# Quick Commands Cheat Sheet

## Setup

```bash
# Activate environment
source .venv/bin/activate

# Navigate to repo root
cd /home/raj_99/Projects/SPIT_Hackathon
```

## Training

```bash
# Train with defaults (ConvNeXt, 10 epochs, lr=1e-4)
python main.py train

# Train with custom model
python main.py train --model linear_head

# Train with custom hyperparams
python main.py train --model convnext_head --epochs 20 --batch_size 4 --lr 5e-4

# Train with larger backbone
python main.py train --backbone_size base --epochs 5

# Train and save to custom location
python main.py train --save_dir /tmp/my_models

# See all training options
python main.py train --help
python -m offroad_training_pipeline.train --help
```

Output locations:
- Model: `offroad_training_pipeline/checkpoints/convnext_head.pth`
- Plots: `offroad_training_pipeline/train_stats/*.png`
- Metrics: `offroad_training_pipeline/train_stats/evaluation_metrics.txt`

## Inference / Testing

```bash
# Test with defaults (ConvNeXt model, test dataset)
python main.py test

# Test with specific model
python main.py test --model linear_head

# Test with custom model path
python main.py test --model_path /tmp/my_models/my_head.pth --model my_head

# Test on validation set instead of test set
python main.py test --data_dir dataset/Offroad_Segmentation_Training_Dataset/val

# Generate more comparison images
python main.py test --num_samples 20

# See all test options
python main.py test --help
```

Output locations:
- Raw masks: `offroad_training_pipeline/predictions/masks/`
- Coloured masks: `offroad_training_pipeline/predictions/masks_color/`
- Comparisons: `offroad_training_pipeline/predictions/comparisons/`
- Metrics: `offroad_training_pipeline/predictions/evaluation_metrics.txt`
- Chart: `offroad_training_pipeline/predictions/per_class_metrics.png`

## Visualisation

```bash
# Colorise segmentation masks
python main.py visualize

# Colorise from custom directory
python main.py visualize --input_dir /path/to/masks

# See options
python main.py visualize --help
```

## Using the Original Scripts

```bash
# Train with original script (saves to new location)
cd Offroad_Segmentation_Scripts
python train_segmentation.py

# Test with original script
python test_segmentation.py --model_path ../offroad_training_pipeline/checkpoints/convnext_head.pth

# Visualise with original script
python visualize.py
```

## Adding a New Model

1. Create model file:
```bash
cat > offroad_training_pipeline/models/my_head.py << 'EOF'
from torch import nn
from offroad_training_pipeline.models.registry import register_model

@register_model("my_head")
class MyHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_w, token_h):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.fc = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.fc(x)
EOF
```

2. Register it:
```bash
# Add to offroad_training_pipeline/models/__init__.py:
echo "import offroad_training_pipeline.models.my_head  # noqa: F401" >> offroad_training_pipeline/models/__init__.py
```

3. Use it:
```bash
python main.py train --model my_head
python main.py test --model my_head
```

## Configuration

Edit defaults in `offroad_training_pipeline/config.py`:

```python
# Change batch size
BATCH_SIZE = 4

# Change learning rate
LEARNING_RATE = 5e-4

# Change number of epochs
NUM_EPOCHS = 20

# Change image size (must be divisible by 14)
IMG_H = 280  # instead of 266
IMG_W = 490  # instead of 476
```

Then run normally (new defaults will be used):
```bash
python main.py train  # uses updated config
```

## Common Tasks

### Compare two models on test set
```bash
# Train model A
python main.py train --model convnext_head --epochs 10 \
  --save_dir offroad_training_pipeline/checkpoints

# Train model B
python main.py train --model linear_head --epochs 10 \
  --save_dir offroad_training_pipeline/checkpoints

# Test both
python main.py test --model convnext_head \
  --output_dir offroad_training_pipeline/predictions/convnext

python main.py test --model linear_head \
  --output_dir offroad_training_pipeline/predictions/linear

# Compare evaluation_metrics.txt in both output dirs
diff offroad_training_pipeline/predictions/convnext/evaluation_metrics.txt \
     offroad_training_pipeline/predictions/linear/evaluation_metrics.txt
```

### Hyperparameter sweep
```bash
# Try different learning rates
for lr in 1e-4 5e-4 1e-3 5e-3; do
  python main.py train --lr $lr \
    --save_dir offroad_training_pipeline/checkpoints \
    --output_dir offroad_training_pipeline/train_stats_lr_$lr
done

# Compare results
ls offroad_training_pipeline/train_stats_lr_*/evaluation_metrics.txt
```

### Debug a single batch
```bash
python -c "
from offroad_training_pipeline.dataset import build_dataloader
from offroad_training_pipeline.config import TRAIN_DIR, BATCH_SIZE

loader = build_dataloader(TRAIN_DIR, BATCH_SIZE, num_workers=0)
imgs, labels = next(iter(loader))
print(f'Batch shape: {imgs.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Image dtype: {imgs.dtype}, range: [{imgs.min():.2f}, {imgs.max():.2f}]')
"
```

### Test a metric function
```bash
python -c "
import torch
from offroad_training_pipeline.metrics import compute_iou
from offroad_training_pipeline.config import NUM_CLASSES

# Create mock data
pred = torch.randn(2, NUM_CLASSES, 64, 64)
target = torch.randint(0, NUM_CLASSES, (2, 64, 64))

# Compute metric
iou = compute_iou(pred, target)
print(f'IoU: {iou:.4f}')
"
```

### Check GPU memory
```bash
python -c "
import torch
from offroad_training_pipeline.config import DEVICE
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print(f'CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
    print(f'CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"
```

## Troubleshooting

### Import errors
```bash
# Make sure you're in repo root
cd /home/raj_99/Projects/SPIT_Hackathon

# Check Python path
python -c "import sys; print(sys.path)"

# Try explicit path
export PYTHONPATH="${PWD}:$PYTHONPATH"
python main.py train --help
```

### File not found
```bash
# Verify dataset structure
tree dataset/ -d

# Verify paths resolve
python -c "
from offroad_training_pipeline.config import TRAIN_DIR, VAL_DIR, TEST_DIR
import os
print('TRAIN_DIR exists:', os.path.exists(TRAIN_DIR))
print('VAL_DIR exists:', os.path.exists(VAL_DIR))
print('TEST_DIR exists:', os.path.exists(TEST_DIR))
"
```

### Out of memory
```bash
# Reduce batch size
python main.py train --batch_size 1

# Use smaller backbone
python main.py train --backbone_size small

# Reduce number of workers
python main.py train --num_workers 0
```

### CUDA not detected
```bash
# Check PyTorch CUDA support
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
"

# If not available, will fall back to CPU (slower)
```

## File Locations Reference

| Purpose | Path |
|---------|------|
| Config | `offroad_training_pipeline/config.py` |
| Training data | `dataset/Offroad_Segmentation_Training_Dataset/train/` |
| Validation data | `dataset/Offroad_Segmentation_Training_Dataset/val/` |
| Test data | `dataset/Offroad_Segmentation_testImages/` |
| Saved models | `offroad_training_pipeline/checkpoints/` |
| Training plots | `offroad_training_pipeline/train_stats/` |
| Predictions | `offroad_training_pipeline/predictions/` |
| Model definitions | `offroad_training_pipeline/models/` |
| Documentation | `offroad_training_pipeline/README.md` |
| Design notes | `offroad_training_pipeline/DESIGN.md` |

## Getting Help

```bash
# See all CLI options
python main.py train --help
python main.py test --help
python main.py visualize --help

# Read the full documentation
cat offroad_training_pipeline/README.md

# Check the design document
cat offroad_training_pipeline/DESIGN.md

# View configuration
cat offroad_training_pipeline/config.py

# Look at a specific module
cat offroad_training_pipeline/metrics.py
```
