# Refactoring Summary

## What Was Done

All code has been **refactored and reorganised** into a modular, extensible structure inside `offroad_training_pipeline/`.

### Original Code
- 3 standalone scripts: `train_segmentation.py`, `test_segmentation.py`, `visualize.py`
- Monolithic files with duplicated logic
- Hardcoded paths and configuration

### Refactored Code Structure

```
offroad_training_pipeline/
├── config.py              # Central config: paths, hyperparams, classes
├── dataset.py             # Data loading, transforms, MaskDataset
├── utils.py               # Image/mask utilities (save, denorm, convert)
├── metrics.py             # IoU, Dice, Pixel Accuracy, evaluate_metrics()
├── visualization.py       # Training plots, comparison images
├── models/
│   ├── registry.py        # Model registry system
│   ├── backbone.py        # DINOv2 loader
│   ├── convnext_head.py   # Default ConvNeXt head
│   └── linear_head.py     # Baseline 1×1 conv head
├── train.py               # Training script (CLI)
├── test.py                # Inference script (CLI)
├── visualize.py           # Mask colorisation script (CLI)
└── README.md              # Full documentation
```

## Key Features

### 1. **Centralised Configuration** (`config.py`)
- All paths resolve from repo root → no more relative path bugs
- All hyperparams in one place → easy to experiment
- Class definitions & colour palette included

### 2. **Model Registry System**
```python
@register_model("my_model")
class MyModel(nn.Module):
    ...

# Use it
python main.py train --model my_model
```
- Add new segmentation heads by just decorating with `@register_model`
- No need to modify training loop

### 3. **DRY Code**
- `MaskDataset`, transforms, metrics, and visualisation functions defined **once**
- Imported everywhere → no duplication
- Changes in one place affect the whole pipeline

### 4. **Modular Scripts**
- `train.py` – flexible CLI with argparse
- `test.py` – inference with per-class metrics
- `visualize.py` – batch mask colorisation
- Each script can be called standalone: `python -m offroad_training_pipeline.train`

### 5. **Unified Entry Point** (`main.py`)
```bash
python main.py train --model convnext_head --epochs 10
python main.py test --model linear_head
python main.py visualize
```

## Path Updates

### Refactored Code
All paths in `offroad_training_pipeline/` resolve from **repo root** (`SPIT_Hackathon/`):
- Datasets → `dataset/`
- Checkpoints → `offroad_training_pipeline/checkpoints/`
- Outputs → `offroad_training_pipeline/train_stats/` & `predictions/`

### Original Scripts
Updated to use **repo root** instead of script directory:
- `Offroad_Segmentation_Scripts/train_segmentation.py` now saves to `offroad_training_pipeline/checkpoints/`
- `Offroad_Segmentation_Scripts/test_segmentation.py` loads from `offroad_training_pipeline/checkpoints/`
- All paths are absolute from repo root

## Usage Examples

### Train the default ConvNeXt head
```bash
python main.py train --model convnext_head --epochs 10
```

### Train a linear baseline
```bash
python main.py train --model linear_head --epochs 20 --batch_size 4
```

### Test / Inference
```bash
python main.py test --model convnext_head \
  --model_path offroad_training_pipeline/checkpoints/convnext_head.pth
```

### Add a new segmentation head
1. Create `offroad_training_pipeline/models/my_head.py`
2. Use `@register_model("my_head")` decorator
3. Import in `offroad_training_pipeline/models/__init__.py`
4. Use: `python main.py train --model my_head`

## File Locations

| Purpose | Location |
|---------|----------|
| Training data | `dataset/Offroad_Segmentation_Training_Dataset/` |
| Test data | `dataset/Offroad_Segmentation_testImages/` |
| Config | `offroad_training_pipeline/config.py` |
| Saved models | `offroad_training_pipeline/checkpoints/` |
| Training history | `offroad_training_pipeline/train_stats/` |
| Predictions | `offroad_training_pipeline/predictions/` |

## Verification

All paths have been verified to exist:
```
✓ TRAIN_DIR exists
✓ VAL_DIR exists
✓ TEST_DIR exists
✓ All imports working
✓ Model registry populated
```

## Next Steps

1. **Train a model:**
   ```bash
   python main.py train --epochs 10
   ```

2. **Run inference:**
   ```bash
   python main.py test
   ```

3. **Experiment with new heads:**
   - Copy `offroad_training_pipeline/models/convnext_head.py` to `my_head.py`
   - Modify architecture
   - Change `@register_model("convnext_head")` → `@register_model("my_head")`
   - Import in `models/__init__.py`
   - Run: `python main.py train --model my_head`

## Documentation

Full documentation available in:
- `offroad_training_pipeline/README.md` – Complete guide with examples
- Each `.py` file has docstrings explaining its purpose
