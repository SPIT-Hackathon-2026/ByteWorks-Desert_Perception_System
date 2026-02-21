"""
Loss functions: Focal Loss + Dice Loss combined.

This combination handles:
  - Severe class imbalance (Obstacle class has very few pixels) → Focal Loss
  - Boundary quality and per-class balance → Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from umixformer_pipeline.config import FOCAL_ALPHA, FOCAL_GAMMA, DICE_SMOOTH, LOSS_WEIGHTS


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.

    Downweights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                 weight: torch.Tensor | None = None, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W) long
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight,
                                  reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Per-class Dice Loss averaged over classes.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth: float = DICE_SMOOTH, num_classes: int = 4,
                 ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Create valid mask
        valid = (targets != self.ignore_index)
        targets_valid = targets.clone()
        targets_valid[~valid] = 0

        # One-hot encode targets
        one_hot = F.one_hot(targets_valid, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Mask invalid pixels
        valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        # Per-class dice
        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of Focal + Dice losses."""

    def __init__(self, num_classes: int = 4, class_weights: torch.Tensor | None = None,
                 focal_weight: float = LOSS_WEIGHTS["focal"],
                 dice_weight: float = LOSS_WEIGHTS["dice"]):
        super().__init__()
        self.focal = FocalLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """Returns dict with individual and combined losses."""
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return {
            "total": total,
            "focal": focal_loss,
            "dice": dice_loss,
        }
