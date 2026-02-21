"""
Segmentation metrics: IoU, Dice, Pixel Accuracy — per-class and mean.
"""

import torch
import numpy as np


def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor,
                             num_classes: int) -> torch.Tensor:
    """Compute confusion matrix.

    Parameters
    ----------
    preds   : (B, C, H, W) logits or (B, H, W) class indices
    targets : (B, H, W) long

    Returns
    -------
    cm : (num_classes, num_classes) — rows=true, cols=predicted
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long,
                     device=preds.device)
    for t in range(num_classes):
        mask = targets == t
        for p in range(num_classes):
            cm[t, p] = (preds[mask] == p).sum()
    return cm


def iou_from_confusion(cm: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-class IoU and mean IoU from confusion matrix."""
    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou = intersection / (union + 1e-8)
    return iou, iou.mean().item()


def dice_from_confusion(cm: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-class Dice and mean Dice from confusion matrix."""
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return dice, dice.mean().item()


def pixel_accuracy_from_confusion(cm: torch.Tensor) -> float:
    """Overall pixel accuracy from confusion matrix."""
    return (cm.diag().sum().float() / (cm.sum().float() + 1e-8)).item()


# ============================================================================
# Convenience functions for training loop
# ============================================================================

def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                num_classes: int) -> float:
    """Mean IoU (scalar)."""
    cm = compute_confusion_matrix(preds, targets, num_classes)
    _, miou = iou_from_confusion(cm)
    return miou


def compute_dice(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int) -> float:
    """Mean Dice (scalar)."""
    cm = compute_confusion_matrix(preds, targets, num_classes)
    _, mdice = dice_from_confusion(cm)
    return mdice


def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Pixel accuracy (scalar)."""
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / (total + 1e-8)


def full_evaluation(preds_all: list[torch.Tensor], targets_all: list[torch.Tensor],
                    num_classes: int, class_names: list[str]) -> dict:
    """Aggregate evaluation over a full dataset.

    Returns detailed metrics dict.
    """
    # Accumulate global confusion matrix
    total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for preds, targets in zip(preds_all, targets_all):
        cm = compute_confusion_matrix(preds.cpu(), targets.cpu(), num_classes)
        total_cm += cm

    per_class_iou, mean_iou = iou_from_confusion(total_cm)
    per_class_dice, mean_dice = dice_from_confusion(total_cm)
    pixel_acc = pixel_accuracy_from_confusion(total_cm)

    results = {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": {class_names[i]: per_class_iou[i].item()
                          for i in range(num_classes)},
        "per_class_dice": {class_names[i]: per_class_dice[i].item()
                           for i in range(num_classes)},
        "confusion_matrix": total_cm.numpy(),
    }
    return results
