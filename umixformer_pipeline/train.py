"""
Training script for U-MixFormer off-road segmentation.

Key improvements over the old pipeline:
  - ConvNeXt encoder with differential LR (encoder 0.1×, decoder 1×)
  - AMP mixed-precision training
  - Gradient accumulation (effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS)
  - Combined Focal + Dice loss for class-imbalanced data
  - Heavy data augmentation via albumentations
  - Cosine annealing with warmup
  - Class-weight computation from training data distribution
  - Per-class IoU/Dice tracking during training

Usage
-----
    python -m umixformer_pipeline.train
    python -m umixformer_pipeline.train --epochs 80 --batch_size 2
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

from umixformer_pipeline.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    ENCODER_LR_MULT,
    GRAD_ACCUM_STEPS,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    TRAIN_DIR,
    VAL_DIR,
    WEIGHT_DECAY,
)
from umixformer_pipeline.dataset import build_train_loader, build_val_loader
from umixformer_pipeline.losses import CombinedLoss
from umixformer_pipeline.metrics import (
    compute_confusion_matrix,
    iou_from_confusion,
    dice_from_confusion,
    pixel_accuracy_from_confusion,
)
from umixformer_pipeline.model import UMixFormerSeg


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train U-MixFormer segmentation")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--encoder_lr_mult", type=float, default=ENCODER_LR_MULT)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    p.add_argument("--val_dir", type=str, default=VAL_DIR)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision")
    p.add_argument("--no_amp", action="store_true", default=False)
    return p.parse_args()


# ============================================================================
# Class weight computation (inverse-frequency weighting)
# ============================================================================

def compute_class_weights(dataloader, num_classes: int,
                          device: torch.device) -> torch.Tensor:
    """Compute class weights from training set based on inverse frequency.

    This is critical for Obstacle class which has very few pixels.
    """
    print("Computing class weights from training data...")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for images, masks in tqdm(dataloader, desc="Scanning class distribution"):
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()

    total = counts.sum()
    freq = counts / total
    # Inverse frequency, clamped to avoid extreme weights
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalise so they sum to num_classes
    # Clamp maximum weight to 10× median to avoid instability
    median_w = weights.median()
    weights = torch.clamp(weights, max=median_w * 10)

    print(f"  Class distribution: {dict(zip(CLASS_NAMES, [f'{f:.4f}' for f in freq.tolist()]))}")
    print(f"  Class weights:      {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights.tolist()]))}")
    return weights.float().to(device)


# ============================================================================
# Cosine LR scheduler with linear warmup
# ============================================================================

def build_scheduler(optimizer, num_epochs: int, steps_per_epoch: int,
                    warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Single epoch: train
# ============================================================================

@torch.no_grad()
def _accumulate_cm(cm_total, logits, targets, num_classes):
    cm = compute_confusion_matrix(logits.detach(), targets, num_classes)
    cm_total += cm.cpu()


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, device, grad_accum_steps, use_amp, epoch,
                    num_classes):
    model.train()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0
    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    num_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for step, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss_dict = criterion(logits, masks)
            loss = loss_dict["total"] / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss_dict["total"].item()
        running_focal += loss_dict["focal"].item()
        running_dice += loss_dict["dice"].item()
        num_batches += 1

        # Accumulate confusion matrix
        _accumulate_cm(cm_total, logits, masks, num_classes)

        pbar.set_postfix({
            "loss": f"{running_loss / num_batches:.4f}",
            "lr": f"{optimizer.param_groups[-1]['lr']:.6f}",
        })

    # Epoch metrics
    avg_loss = running_loss / num_batches
    avg_focal = running_focal / num_batches
    avg_dice = running_dice / num_batches
    per_iou, miou = iou_from_confusion(cm_total)
    per_dice, mdice = dice_from_confusion(cm_total)
    pix_acc = pixel_accuracy_from_confusion(cm_total)

    metrics = {
        "loss": avg_loss,
        "focal_loss": avg_focal,
        "dice_loss": avg_dice,
        "mean_iou": miou,
        "mean_dice": mdice,
        "pixel_acc": pix_acc,
        "per_class_iou": {CLASS_NAMES[i]: per_iou[i].item()
                          for i in range(num_classes)},
        "per_class_dice": {CLASS_NAMES[i]: per_dice[i].item()
                           for i in range(num_classes)},
    }
    return metrics


# ============================================================================
# Single epoch: validate
# ============================================================================

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch,
                       num_classes, use_amp):
    model.eval()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0
    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Val   Epoch {epoch}", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss_dict = criterion(logits, masks)

        running_loss += loss_dict["total"].item()
        running_focal += loss_dict["focal"].item()
        running_dice += loss_dict["dice"].item()
        num_batches += 1

        _accumulate_cm(cm_total, logits, masks, num_classes)

    avg_loss = running_loss / num_batches
    avg_focal = running_focal / num_batches
    avg_dice = running_dice / num_batches
    per_iou, miou = iou_from_confusion(cm_total)
    per_dice, mdice = dice_from_confusion(cm_total)
    pix_acc = pixel_accuracy_from_confusion(cm_total)

    metrics = {
        "loss": avg_loss,
        "focal_loss": avg_focal,
        "dice_loss": avg_dice,
        "mean_iou": miou,
        "mean_dice": mdice,
        "pixel_acc": pix_acc,
        "per_class_iou": {CLASS_NAMES[i]: per_iou[i].item()
                          for i in range(num_classes)},
        "per_class_dice": {CLASS_NAMES[i]: per_dice[i].item()
                           for i in range(num_classes)},
    }
    return metrics


# ============================================================================
# Main training loop
# ============================================================================

def main():
    args = parse_args()
    use_amp = args.amp and not args.no_amp and torch.cuda.is_available()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  U-MixFormer Off-Road Segmentation Training")
    print(f"{'='*60}")
    print(f"  Device:        {DEVICE}")
    print(f"  Image size:    {args.img_size}×{args.img_size}")
    print(f"  Batch size:    {args.batch_size} (×{args.grad_accum} accum = {args.batch_size * args.grad_accum} effective)")
    print(f"  LR:            {args.lr} (encoder {args.lr * args.encoder_lr_mult})")
    print(f"  Epochs:        {args.epochs}")
    print(f"  AMP:           {use_amp}")
    print(f"  Train dir:     {args.train_dir}")
    print(f"  Val dir:       {args.val_dir}")
    print(f"{'='*60}\n")

    # ---- Data ----
    train_loader = build_train_loader(args.train_dir, args.batch_size, args.num_workers)
    val_loader = build_val_loader(args.val_dir, args.batch_size, args.num_workers)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    # ---- Class weights ----
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, DEVICE)

    # ---- Model ----
    model = UMixFormerSeg(pretrained_encoder=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {total_params:.1f}M total, {trainable_params:.1f}M trainable\n")

    # ---- Optimizer with differential LR ----
    param_groups = model.get_param_groups(encoder_lr_mult=args.encoder_lr_mult)
    optimizer = torch.optim.AdamW([
        {"params": param_groups[0]["params"],
         "lr": args.lr * param_groups[0]["lr_mult"]},
        {"params": param_groups[1]["params"],
         "lr": args.lr * param_groups[1]["lr_mult"]},
    ], weight_decay=args.weight_decay)

    # ---- Scheduler: cosine with warmup ----
    steps_per_epoch = len(train_loader) // args.grad_accum + 1
    scheduler = build_scheduler(optimizer, args.epochs, steps_per_epoch,
                                warmup_epochs=5)

    # ---- Loss ----
    criterion = CombinedLoss(num_classes=NUM_CLASSES,
                             class_weights=class_weights).to(DEVICE)

    # ---- AMP scaler ----
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best val IoU: {best_val_iou:.4f}")

    # ---- History tracking ----
    history = {
        "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "train_dice": [], "val_dice": [],
        "train_pix_acc": [], "val_pix_acc": [],
        "lr": [],
        "per_class_val_iou": [],
    }

    # ---- Training ----
    print("Starting training...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, DEVICE, args.grad_accum, use_amp, epoch + 1, NUM_CLASSES
        )

        val_metrics = validate_one_epoch(
            model, val_loader, criterion, DEVICE, epoch + 1,
            NUM_CLASSES, use_amp
        )

        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["mean_iou"])
        history["val_iou"].append(val_metrics["mean_iou"])
        history["train_dice"].append(train_metrics["mean_dice"])
        history["val_dice"].append(val_metrics["mean_dice"])
        history["train_pix_acc"].append(train_metrics["pixel_acc"])
        history["val_pix_acc"].append(val_metrics["pixel_acc"])
        history["lr"].append(optimizer.param_groups[-1]["lr"])
        history["per_class_val_iou"].append(val_metrics["per_class_iou"])

        # Print summary
        print(f"\nEpoch {epoch + 1}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"  Train — Loss: {train_metrics['loss']:.4f}  "
              f"mIoU: {train_metrics['mean_iou']:.4f}  "
              f"Dice: {train_metrics['mean_dice']:.4f}  "
              f"Acc: {train_metrics['pixel_acc']:.4f}")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f}  "
              f"mIoU: {val_metrics['mean_iou']:.4f}  "
              f"Dice: {val_metrics['mean_dice']:.4f}  "
              f"Acc: {val_metrics['pixel_acc']:.4f}")
        print(f"  Val per-class IoU: ", end="")
        for name, v in val_metrics["per_class_iou"].items():
            print(f"{name}={v:.4f}  ", end="")
        print()

        # Save best model
        is_best = val_metrics["mean_iou"] > best_val_iou
        if is_best:
            best_val_iou = val_metrics["mean_iou"]
            print(f"  ★ New best val IoU: {best_val_iou:.4f}")

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "umixformer_latest.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, "umixformer_best.pth"))

        # Save periodic checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(args.save_dir,
                                          f"umixformer_epoch{epoch+1}.pth"))

    # ---- Save final results ----
    print(f"\n{'='*60}")
    print(f"Training complete. Best Val mIoU: {best_val_iou:.4f}")
    print(f"{'='*60}")

    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    # Save final metrics summary
    summary_path = os.path.join(args.output_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("U-MixFormer Off-Road Segmentation — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Validation mIoU: {best_val_iou:.4f}\n")
        f.write(f"Final Train mIoU:     {history['train_iou'][-1]:.4f}\n")
        f.write(f"Final Val mIoU:       {history['val_iou'][-1]:.4f}\n")
        f.write(f"Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Val PixAcc:     {history['val_pix_acc'][-1]:.4f}\n\n")
        f.write("Per-class Val IoU (best epoch):\n")
        best_epoch_idx = int(np.argmax(history["val_iou"]))
        best_per_class = history["per_class_val_iou"][best_epoch_idx]
        for name, v in best_per_class.items():
            f.write(f"  {name:12s}: {v:.4f}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
