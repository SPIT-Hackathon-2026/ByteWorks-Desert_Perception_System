"""
Segmentation metrics: IoU, Dice, Pixel Accuracy, and a combined evaluator.
"""

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from offroad_training_pipeline.config import NUM_CLASSES


# ============================================================================
# Per-batch metric functions
# ============================================================================

def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
    return_per_class: bool = False,
    class_filter: list | None = None,
):
    """Compute (mean) Intersection-over-Union.

    Parameters
    ----------
    pred : (B, C, H, W) logits
    target : (B, H, W) class ids
    return_per_class : bool
        If *True* also return the per-class list.
    class_filter : list[int] | None
        Evaluate only these class indices.  *None* → all classes.
    """
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    # Drop pixels with the ignore label (e.g. CV-class pixels set to 255)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    classes = class_filter if class_filter is not None else list(range(num_classes))

    iou_per_class = []
    for cid in classes:
        p = pred == cid
        t = target == cid
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou_per_class.append(float("nan") if union == 0 else (inter / union).cpu().item())

    mean = float(np.nanmean(iou_per_class))
    if return_per_class:
        return mean, iou_per_class
    return mean


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = 1e-6,
    return_per_class: bool = False,
    ignore_index: int = 255,
    class_filter: list | None = None,
):
    """Compute (mean) Dice / F1 score."""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    classes = class_filter if class_filter is not None else list(range(num_classes))

    dice_per_class = []
    for cid in classes:
        p = pred == cid
        t = target == cid
        inter = (p & t).sum().float()
        score = (2.0 * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dice_per_class.append(score.cpu().item())

    mean = float(np.mean(dice_per_class))
    if return_per_class:
        return mean, dice_per_class
    return mean


def compute_pixel_accuracy(
    pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255
) -> float:
    """Compute pixel-level accuracy (ignoring masked pixels)."""
    pred_cls = torch.argmax(pred, dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return 1.0
    return (pred_cls[valid] == target[valid]).float().mean().cpu().item()


# ============================================================================
# Full-dataset evaluator
# ============================================================================

def evaluate_metrics(
    model,
    backbone,
    data_loader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    show_progress: bool = True,
    return_per_class: bool = False,
):
    """Run the segmentation head on an entire dataloader and return metrics.

    Returns
    -------
    mean_iou, mean_dice, mean_pixel_acc : float
    (optionally) avg_class_iou, avg_class_dice : list  (when *return_per_class*)
    """
    iou_scores, dice_scores, pixel_accs = [], [], []
    all_class_iou, all_class_dice = [], []

    model.eval()
    loader = (
        tqdm(data_loader, desc="Evaluating", leave=False, unit="batch")
        if show_progress
        else data_loader
    )

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)

            tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(tokens)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(1).long()

            iou, c_iou = compute_iou(outputs, labels, num_classes, return_per_class=True)
            dice, c_dice = compute_dice(outputs, labels, num_classes, return_per_class=True)
            pacc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accs.append(pacc)
            all_class_iou.append(c_iou)
            all_class_dice.append(c_dice)

    model.train()

    mean_iou = float(np.nanmean(iou_scores))
    mean_dice = float(np.nanmean(dice_scores))
    mean_pacc = float(np.mean(pixel_accs))

    if return_per_class:
        avg_c_iou = np.nanmean(all_class_iou, axis=0).tolist()
        avg_c_dice = np.nanmean(all_class_dice, axis=0).tolist()
        return mean_iou, mean_dice, mean_pacc, avg_c_iou, avg_c_dice

    return mean_iou, mean_dice, mean_pacc
