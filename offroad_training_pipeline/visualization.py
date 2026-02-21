"""
Plotting / visualisation helpers for training curves and prediction overlays.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from offroad_training_pipeline.config import (
    CLASS_NAMES,
    COLOR_PALETTE,
    NUM_CLASSES,
)
from offroad_training_pipeline.utils import denormalize_image, mask_to_color


# ============================================================================
# Training-curve plots
# ============================================================================

def save_training_plots(history: dict, output_dir: str) -> None:
    """Generate and save all standard training metric plots."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Loss + Pixel Accuracy -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_pixel_acc"], label="train")
    axes[1].plot(history["val_pixel_acc"], label="val")
    axes[1].set(title="Pixel Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close(fig)

    # ---- IoU curves -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_iou"], label="Train IoU")
    axes[0].set(title="Train IoU vs Epoch", xlabel="Epoch", ylabel="IoU")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_iou"], label="Val IoU")
    axes[1].set(title="Val IoU vs Epoch", xlabel="Epoch", ylabel="IoU")
    axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iou_curves.png"))
    plt.close(fig)

    # ---- Dice curves ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_dice"], label="Train Dice")
    axes[0].set(title="Train Dice vs Epoch", xlabel="Epoch", ylabel="Dice Score")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_dice"], label="Val Dice")
    axes[1].set(title="Val Dice vs Epoch", xlabel="Epoch", ylabel="Dice Score")
    axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dice_curves.png"))
    plt.close(fig)

    # ---- Combined 2×2 overview -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, key, title in [
        (axes[0, 0], "loss", "Loss vs Epoch"),
        (axes[0, 1], "iou", "IoU vs Epoch"),
        (axes[1, 0], "dice", "Dice Score vs Epoch"),
        (axes[1, 1], "pixel_acc", "Pixel Accuracy vs Epoch"),
    ]:
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set(title=title, xlabel="Epoch", ylabel=key.replace("_", " ").title())
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_metrics_curves.png"))
    plt.close(fig)

    print(f"Saved training plots → {output_dir}/")


# ============================================================================
# History text dump
# ============================================================================

def save_history_to_file(history: dict, output_dir: str) -> None:
    """Write a human-readable summary of training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(filepath, "w") as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        for tag in ["train_loss", "val_loss", "train_iou", "val_iou",
                     "train_dice", "val_dice", "train_pixel_acc", "val_pixel_acc"]:
            label = tag.replace("_", " ").title()
            f.write(f"  {label:<24}: {history[tag][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        for tag, direction in [("val_iou", "max"), ("val_dice", "max"),
                               ("val_pixel_acc", "max"), ("val_loss", "min")]:
            fn = max if direction == "max" else min
            idx_fn = np.argmax if direction == "max" else np.argmin
            label = tag.replace("_", " ").title()
            f.write(f"  Best {label:<20}: {fn(history[tag]):.4f} (Epoch {int(idx_fn(history[tag])) + 1})\n")
        f.write("=" * 50 + "\n\n")

        headers = ["Epoch", "Train Loss", "Val Loss", "Train IoU", "Val IoU",
                    "Train Dice", "Val Dice", "Train Acc", "Val Acc"]
        fmt = "{:<8}" + "{:<12}" * 8 + "\n"
        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        f.write(fmt.format(*headers))
        f.write("-" * 100 + "\n")

        for i in range(len(history["train_loss"])):
            f.write(fmt.format(
                i + 1,
                f"{history['train_loss'][i]:.4f}",
                f"{history['val_loss'][i]:.4f}",
                f"{history['train_iou'][i]:.4f}",
                f"{history['val_iou'][i]:.4f}",
                f"{history['train_dice'][i]:.4f}",
                f"{history['val_dice'][i]:.4f}",
                f"{history['train_pixel_acc'][i]:.4f}",
                f"{history['val_pixel_acc'][i]:.4f}",
            ))

    print(f"Saved evaluation metrics → {filepath}")


# ============================================================================
# Prediction comparison image
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id=""):
    """Save a 3-panel figure: input | ground truth | prediction."""
    img = denormalize_image(img_tensor)
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(axes, [img, gt_color, pred_color],
                              ["Input Image", "Ground Truth", "Prediction"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    if data_id:
        plt.suptitle(f"Sample: {data_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Per-class bar chart
# ============================================================================

def save_metrics_summary(results: dict, output_dir: str) -> None:
    """Save evaluation results to text + per-class IoU bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Text file
    filepath = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(filepath, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-Class IoU:\n" + "-" * 40 + "\n")
        for name, iou in zip(CLASS_NAMES, results["class_iou"]):
            f.write(f"  {name:<20}: {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}: N/A")
            f.write("\n")
    print(f"Saved evaluation metrics → {filepath}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = [0 if np.isnan(v) else v for v in results["class_iou"]]
    ax.bar(range(NUM_CLASSES), valid,
           color=[COLOR_PALETTE[i] / 255 for i in range(NUM_CLASSES)],
           edgecolor="black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set(ylabel="IoU", title=f"Per-Class IoU (Mean: {results['mean_iou']:.4f})", ylim=(0, 1))
    ax.axhline(results["mean_iou"], color="red", ls="--", label="Mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-class chart → {output_dir}/per_class_metrics.png")
