"""
Compare model-only vs model+preprocessing evaluation.

Runs the U-MixFormer model on the same dataset twice:
  1. Raw images (standard pipeline)
  2. Preprocessed images (dehaze + histogram equalisation applied first)

Prints a side-by-side comparison table and saves it as CSV + text.

Usage::

    python -m umixformer_pipeline.compare_preproc
    python -m umixformer_pipeline.compare_preproc --split test --checkpoint path/to/best.pth
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from umixformer_pipeline.config import (
    CLASS_NAMES,
    DEVICE,
    IMG_SIZE,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    PREDICTIONS_DIR,
    TEST_DIR,
    VAL_DIR,
)
from umixformer_pipeline.dataset import (
    OffroadSegDataset,
    get_val_augmentations,
    build_test_loader,
)
from umixformer_pipeline.metrics import (
    compute_confusion_matrix,
    iou_from_confusion,
    dice_from_confusion,
    pixel_accuracy_from_confusion,
)
from umixformer_pipeline.model import UMixFormerSeg
from umixformer_pipeline.preprocess import preprocess_image
from umixformer_pipeline.utils import convert_mask

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Preprocessed dataset  (applies dehaze + histeq BEFORE augmentation)
# ============================================================================

class PreprocessedDataset(OffroadSegDataset):
    """Same as OffroadSegDataset but applies image preprocessing first."""

    def __getitem__(self, idx):
        fname = self.ids[idx]
        image = np.array(
            Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        )

        # >>> Apply preprocessing (dehaze + histogram equalisation) <<<
        image = preprocess_image(image)  # uint8 → uint8

        mask_pil = Image.open(os.path.join(self.mask_dir, fname))
        mask = np.array(convert_mask(mask_pil), dtype=np.int64)

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        mask = mask.long()
        if self.return_filename:
            return image, mask, fname
        return image, mask


# ============================================================================
# Evaluation helper
# ============================================================================

@torch.no_grad()
def _evaluate_loader(model, dataloader, device) -> dict:
    """Run evaluation, return metrics dict."""
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        images, masks = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)

        preds = logits.argmax(dim=1)
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(
                preds.unsqueeze(1).float(), size=masks.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()

        cm += compute_confusion_matrix(preds, masks, NUM_CLASSES).cpu()

    per_iou, miou = iou_from_confusion(cm)
    per_dice, mdice = dice_from_confusion(cm)
    pix_acc = pixel_accuracy_from_confusion(cm)

    return {
        "mean_iou": miou,
        "mean_dice": mdice,
        "pixel_accuracy": pix_acc,
        "per_class_iou": {CLASS_NAMES[i]: per_iou[i].item()
                          for i in range(NUM_CLASSES)},
        "per_class_dice": {CLASS_NAMES[i]: per_dice[i].item()
                           for i in range(NUM_CLASSES)},
    }


# ============================================================================
# Comparison table printer
# ============================================================================

def _print_comparison(raw_m: dict, pre_m: dict, split: str, out_dir: str):
    """Pretty-print and save side-by-side comparison."""

    sep = "=" * 78
    hdr = f"  MODEL vs MODEL + PREPROCESSING — {split.upper()} set"

    lines = []
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    lines.append(f"  {'Metric':<22} {'Model Only':>14} {'Model+Preproc':>14} {'Δ':>10}")
    lines.append(f"  {'-'*62}")

    def row(name, v1, v2):
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {name:<22} {v1:>14.4f} {v2:>14.4f} {sign}{delta:>9.4f}")

    row("Mean IoU", raw_m["mean_iou"], pre_m["mean_iou"])
    row("Mean Dice", raw_m["mean_dice"], pre_m["mean_dice"])
    row("Pixel Accuracy", raw_m["pixel_accuracy"], pre_m["pixel_accuracy"])

    lines.append(f"\n  {'Per-Class IoU':}")
    lines.append(f"  {'-'*62}")
    for cls in CLASS_NAMES:
        row(f"  {cls}", raw_m["per_class_iou"][cls], pre_m["per_class_iou"][cls])

    lines.append(f"\n  {'Per-Class Dice':}")
    lines.append(f"  {'-'*62}")
    for cls in CLASS_NAMES:
        row(f"  {cls}", raw_m["per_class_dice"][cls], pre_m["per_class_dice"][cls])

    lines.append(sep)

    # Verdict
    iou_delta = pre_m["mean_iou"] - raw_m["mean_iou"]
    if iou_delta > 0.005:
        verdict = "✅  Preprocessing IMPROVES segmentation"
    elif iou_delta < -0.005:
        verdict = "❌  Preprocessing HURTS segmentation"
    else:
        verdict = "➖  Preprocessing has NEGLIGIBLE effect"
    lines.append(f"  {verdict}  (ΔmIoU = {iou_delta:+.4f})")
    lines.append(sep)

    txt = "\n".join(lines)
    print(txt)

    # Save text
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"comparison_{split}.txt"), "w") as f:
        f.write(txt + "\n")

    # Save CSV
    csv_path = os.path.join(out_dir, f"comparison_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Model_Only", "Model+Preproc", "Delta"])
        w.writerow(["Mean_IoU", f"{raw_m['mean_iou']:.4f}",
                     f"{pre_m['mean_iou']:.4f}",
                     f"{pre_m['mean_iou'] - raw_m['mean_iou']:+.4f}"])
        w.writerow(["Mean_Dice", f"{raw_m['mean_dice']:.4f}",
                     f"{pre_m['mean_dice']:.4f}",
                     f"{pre_m['mean_dice'] - raw_m['mean_dice']:+.4f}"])
        w.writerow(["Pixel_Accuracy", f"{raw_m['pixel_accuracy']:.4f}",
                     f"{pre_m['pixel_accuracy']:.4f}",
                     f"{pre_m['pixel_accuracy'] - raw_m['pixel_accuracy']:+.4f}"])
        for cls in CLASS_NAMES:
            w.writerow([f"IoU_{cls}",
                         f"{raw_m['per_class_iou'][cls]:.4f}",
                         f"{pre_m['per_class_iou'][cls]:.4f}",
                         f"{pre_m['per_class_iou'][cls] - raw_m['per_class_iou'][cls]:+.4f}"])
        for cls in CLASS_NAMES:
            w.writerow([f"Dice_{cls}",
                         f"{raw_m['per_class_dice'][cls]:.4f}",
                         f"{pre_m['per_class_dice'][cls]:.4f}",
                         f"{pre_m['per_class_dice'][cls] - raw_m['per_class_dice'][cls]:+.4f}"])
    print(f"  CSV → {csv_path}")

    # Bar chart
    _comparison_chart(raw_m, pre_m, split, out_dir)


def _comparison_chart(raw_m: dict, pre_m: dict, split: str, out_dir: str):
    """Side-by-side grouped bar chart."""
    metrics = ["Mean IoU", "Mean Dice", "Pixel Acc"]
    raw_vals = [raw_m["mean_iou"], raw_m["mean_dice"], raw_m["pixel_accuracy"]]
    pre_vals = [pre_m["mean_iou"], pre_m["mean_dice"], pre_m["pixel_accuracy"]]

    # Add per-class IoU
    for cls in CLASS_NAMES:
        metrics.append(f"IoU {cls}")
        raw_vals.append(raw_m["per_class_iou"][cls])
        pre_vals.append(pre_m["per_class_iou"][cls])

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, raw_vals, width, label="Model Only",
                   color="#4488cc", edgecolor="black", linewidth=0.4)
    bars2 = ax.bar(x + width / 2, pre_vals, width, label="Model + Preproc",
                   color="#44cc88", edgecolor="black", linewidth=0.4)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Model vs Model+Preprocessing — {split.upper()}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, f"comparison_chart_{split}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {chart_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare model vs model+preprocessing")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(MODEL_SAVE_DIR, "umixformer_best.pth"))
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PREDICTIONS_DIR, "comparison"))
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    # Load model
    model = UMixFormerSeg(pretrained_encoder=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    data_dir = TEST_DIR if args.split == "test" else VAL_DIR
    aug = get_val_augmentations()

    # --- Run 1: Raw images ---
    print("\n[1/2] Evaluating on RAW images …")
    raw_ds = OffroadSegDataset(data_dir, augmentations=aug, return_filename=True)
    from torch.utils.data import DataLoader
    raw_loader = DataLoader(raw_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    t0 = time.time()
    raw_metrics = _evaluate_loader(model, raw_loader, device)
    raw_time = time.time() - t0
    print(f"  Done in {raw_time:.1f}s")

    # --- Run 2: Preprocessed images ---
    print("[2/2] Evaluating on PREPROCESSED images (dehaze + histeq) …")
    pre_ds = PreprocessedDataset(data_dir, augmentations=aug, return_filename=True)
    pre_loader = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)  # num_workers=0 because preprocess is heavy
    t0 = time.time()
    pre_metrics = _evaluate_loader(model, pre_loader, device)
    pre_time = time.time() - t0
    print(f"  Done in {pre_time:.1f}s")

    # --- Print comparison ---
    _print_comparison(raw_metrics, pre_metrics, args.split, args.output_dir)


if __name__ == "__main__":
    main()
