"""
Visualization utilities: training plots, prediction grids, metric tables.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from umixformer_pipeline.config import CLASS_NAMES, OUTPUT_DIR


def plot_training_history(history_path: str = None, output_dir: str = None):
    """Generate training plots from saved history JSON."""
    if history_path is None:
        history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if output_dir is None:
        output_dir = OUTPUT_DIR

    with open(history_path) as f:
        history = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # ---- Loss curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # ---- IoU curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history["train_iou"], "b-", linewidth=2, label="Train mIoU")
    ax.plot(epochs, history["val_iou"], "r-", linewidth=2, label="Val mIoU")
    best_idx = int(np.argmax(history["val_iou"]))
    ax.axvline(x=best_idx + 1, color="green", linestyle="--", alpha=0.7,
               label=f"Best: {history['val_iou'][best_idx]:.4f} (epoch {best_idx+1})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean IoU", fontsize=12)
    ax.set_title("Training & Validation Mean IoU", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iou_curves.png"), dpi=150)
    plt.close(fig)

    # ---- Dice curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history["train_dice"], "b-", linewidth=2, label="Train Dice")
    ax.plot(epochs, history["val_dice"], "r-", linewidth=2, label="Val Dice")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Dice", fontsize=12)
    ax.set_title("Training & Validation Mean Dice", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dice_curves.png"), dpi=150)
    plt.close(fig)

    # ---- Per-class Val IoU over epochs ----
    if "per_class_val_iou" in history and history["per_class_val_iou"]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        colors = ["#2ecc71", "#e67e22", "#e74c3c", "#3498db"]
        for i, name in enumerate(CLASS_NAMES):
            vals = [ep_dict[name] for ep_dict in history["per_class_val_iou"]]
            ax.plot(epochs, vals, linewidth=2, label=name, color=colors[i])
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("IoU", fontsize=12)
        ax.set_title("Per-Class Validation IoU Over Training", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_class_iou.png"), dpi=150)
        plt.close(fig)

    # ---- LR schedule ----
    if "lr" in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(epochs, history["lr"], "purple", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

    # ---- Pixel accuracy ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history["train_pix_acc"], "b-", linewidth=2, label="Train Acc")
    ax.plot(epochs, history["val_pix_acc"], "r-", linewidth=2, label="Val Acc")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Pixel Accuracy", fontsize=12)
    ax.set_title("Training & Validation Pixel Accuracy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pixel_accuracy.png"), dpi=150)
    plt.close(fig)

    print(f"Training plots saved to {output_dir}")


def plot_confusion_matrix(cm: np.ndarray, output_dir: str = None):
    """Plot normalized confusion matrix."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_dir}")


if __name__ == "__main__":
    plot_training_history()
