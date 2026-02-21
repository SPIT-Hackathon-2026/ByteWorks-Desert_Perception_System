"""
Training script for the off-road segmentation pipeline.

Usage
-----
    python -m offroad_training_pipeline.train                          # defaults (hybrid_head)
    python -m offroad_training_pipeline.train --model convnext_head    # explicit head
    python -m offroad_training_pipeline.train --model linear_head --epochs 20
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from offroad_training_pipeline.config import (
    BATCH_SIZE,
    BACKBONE_SIZE,
    CLASS_NAMES,
    DEVICE,
    IMG_H,
    IMG_W,
    LEARNING_RATE,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PATCH_SIZE,
    TRAIN_DIR,
    VAL_DIR,
    WEIGHT_DECAY,
)
from offroad_training_pipeline.dataset import build_dataloader
from offroad_training_pipeline.metrics import compute_iou, compute_dice, compute_pixel_accuracy
from offroad_training_pipeline.models import build_model, load_backbone
from offroad_training_pipeline.models.backbone import get_embedding_dim
from offroad_training_pipeline.visualization import save_history_to_file, save_training_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation head on DINOv2 features")
    parser.add_argument("--model", type=str, default="segformer_head",
                        help="Registered model name (e.g. segformer_head, hybrid_head, convnext_head)")
    parser.add_argument("--backbone_size", type=str, default=BACKBONE_SIZE,
                        help="DINOv2 backbone size: small | base | large | giant")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data
    train_loader = build_dataloader(args.train_dir, args.batch_size,
                                    shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(args.val_dir, args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    print(f"Training samples : {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # -------------------------------------------------------------- backbone
    backbone = load_backbone(args.backbone_size, device)

    sample_img, _ = next(iter(train_loader))
    n_embedding = get_embedding_dim(backbone, sample_img[:1])
    print(f"Embedding dim: {n_embedding}")

    token_w = IMG_W // PATCH_SIZE
    token_h = IMG_H // PATCH_SIZE

    # ----------------------------------------------------------------- model
    classifier = build_model(
        args.model,
        in_channels=n_embedding,
        out_channels=NUM_CLASSES,
        token_w=token_w,
        token_h=token_h,
    ).to(device)
    print(f"Segmentation head: {args.model}")
    param_count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Trainable params : {param_count:,}")

    # --------------------------------------------------------- loss / optim
    # 3 super-classes: Driveable, Obstacle, Sky
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print("Computing class weights from training data…")
    class_counts = torch.zeros(NUM_CLASSES)
    for _, masks in tqdm(train_loader, desc="Scanning class distribution", leave=False):
        masks_long = (masks.squeeze(1) * 255).long() if masks.max() <= 1 else masks.squeeze(1).long()
        for c in range(NUM_CLASSES):
            class_counts[c] += (masks_long == c).sum().item()
    # Inverse-frequency weights
    total_px = class_counts.sum().item()
    class_weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        count = max(class_counts[c].item(), 1)
        class_weights[c] = total_px / (NUM_CLASSES * count)
    # Normalise so weights sum to NUM_CLASSES
    w_sum = class_weights.sum().item()
    if w_sum > 0:
        class_weights *= NUM_CLASSES / w_sum
    print(f"Class weights: { {CLASS_NAMES[c]: round(class_weights[c].item(), 4) for c in range(NUM_CLASSES)} }")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ------------------------------------------------------------- history
    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_iou", "val_iou",
        "train_dice", "val_dice",
        "train_pixel_acc", "val_pixel_acc",
    ]}

    best_val_iou = 0.0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------------------------------------------- training loop
    print(f"\nStarting training for {args.epochs} epochs…")
    print(f"Optimizer : AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"Scheduler : OneCycleLR (10% warmup + cosine)")
    print(f"AMP       : {'enabled' if use_amp else 'disabled'}")
    print("=" * 80)

    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # ---- train ----
        classifier.train()
        train_losses, train_ious, train_dices, train_accs = [], [], [], []
        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = classifier(tokens)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                labels_sq = labels.squeeze(1).long()
                loss = loss_fn(outputs, labels_sq)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            train_losses.append(loss.item())

            with torch.no_grad():
                train_ious.append(compute_iou(outputs, labels_sq, NUM_CLASSES))
                train_dices.append(compute_dice(outputs, labels_sq, NUM_CLASSES))
                train_accs.append(compute_pixel_accuracy(outputs, labels_sq))

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{train_ious[-1]:.3f}",
            )

        # ---- validate (metrics computed inline — NO second "Evaluating" pass) ----
        classifier.eval()
        val_losses, val_ious, val_dices, val_accs = [], [], [], []
        val_pbar = tqdm(val_loader,
                        desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = classifier(tokens)
                    outputs = F.interpolate(logits, size=imgs.shape[2:],
                                            mode="bilinear", align_corners=False)
                    labels_sq = labels.squeeze(1).long()
                    loss = loss_fn(outputs, labels_sq)

                val_losses.append(loss.item())
                val_ious.append(compute_iou(outputs, labels_sq, NUM_CLASSES))
                val_dices.append(compute_dice(outputs, labels_sq, NUM_CLASSES))
                val_accs.append(compute_pixel_accuracy(outputs, labels_sq))

                val_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    iou=f"{val_ious[-1]:.3f}",
                )

        # ---- aggregate epoch metrics ----
        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss = float(np.mean(val_losses))
        train_iou = float(np.nanmean(train_ious))
        val_iou = float(np.nanmean(val_ious))
        train_dice = float(np.nanmean(train_dices))
        val_dice = float(np.nanmean(val_dices))
        train_pacc = float(np.mean(train_accs))
        val_pacc = float(np.mean(val_accs))

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_pixel_acc"].append(train_pacc)
        history["val_pixel_acc"].append(val_pacc)

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            os.makedirs(args.save_dir, exist_ok=True)
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(classifier.state_dict(), best_path)

        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(
            loss=f"{epoch_train_loss:.3f}",
            v_loss=f"{epoch_val_loss:.3f}",
            v_iou=f"{val_iou:.3f}",
            v_acc=f"{val_pacc:.3f}",
            lr=f"{cur_lr:.1e}",
        )

    # ---------------------------------------------------------- save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    save_training_plots(history, args.output_dir)
    save_history_to_file(history, args.output_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{args.model}.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"\nSaved final model → {model_path}")
    if best_val_iou > 0:
        print(f"Saved best model  → {best_path}  (val_iou={best_val_iou:.4f})")

    # ---- summary ----
    print(f"\nFinal evaluation ({NUM_CLASSES} classes: {CLASS_NAMES}):")
    print(f"  Val Loss     : {history['val_loss'][-1]:.4f}")
    print(f"  Val IoU      : {history['val_iou'][-1]:.4f}")
    print(f"  Val Dice     : {history['val_dice'][-1]:.4f}")
    print(f"  Val Accuracy : {history['val_pixel_acc'][-1]:.4f}")
    print(f"  Best Val IoU : {best_val_iou:.4f}")
    print("\nTraining complete! ✓")


if __name__ == "__main__":
    main()
