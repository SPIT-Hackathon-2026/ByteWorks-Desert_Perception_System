"""
Inference / test script for the off-road segmentation pipeline.

Runs a trained segmentation head on a test (or val) set and saves:
* raw prediction masks  (class-id PNGs)
* coloured prediction masks  (RGB PNGs)
* side-by-side comparison images
* per-class IoU chart & text summary

Usage
-----
    python -m offroad_training_pipeline.test
    python -m offroad_training_pipeline.test --model segformer_head
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from offroad_training_pipeline.config import (
    BACKBONE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    IMG_H,
    IMG_W,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    PATCH_SIZE,
    PREDICTIONS_DIR,
    TEST_DIR,
)
from offroad_training_pipeline.dataset import build_dataloader
from offroad_training_pipeline.metrics import (
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
)
from offroad_training_pipeline.models import build_model, load_backbone
from offroad_training_pipeline.models.backbone import get_embedding_dim
from offroad_training_pipeline.utils import denormalize_image, mask_to_color
from offroad_training_pipeline.visualization import save_prediction_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation inference & evaluation")
    parser.add_argument("--model", type=str, default="segformer_head",
                        help="Registered model name")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained .pth weights (auto-detects _best.pth)")
    parser.add_argument("--backbone_size", type=str, default=BACKBONE_SIZE)
    parser.add_argument("--data_dir", type=str, default=TEST_DIR,
                        help="Directory with Color_Images/ and Segmentation/")
    parser.add_argument("--output_dir", type=str, default=PREDICTIONS_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of comparison visualisations to save")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    # Auto-detect best checkpoint
    if args.model_path is None:
        best = os.path.join(MODEL_SAVE_DIR, f"{args.model}_best.pth")
        final = os.path.join(MODEL_SAVE_DIR, f"{args.model}.pth")
        args.model_path = best if os.path.exists(best) else final
    return args


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------- data
    val_loader = build_dataloader(
        args.data_dir, args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        return_filename=True,
    )
    print(f"Loaded {len(val_loader.dataset)} samples from {args.data_dir}")

    # ----------------------------------------------------------- backbone
    backbone = load_backbone(args.backbone_size, device)

    sample_img, _, _ = val_loader.dataset[0]
    n_embedding = get_embedding_dim(backbone, sample_img.unsqueeze(0))
    print(f"Embedding dim: {n_embedding}")

    token_w = IMG_W // PATCH_SIZE
    token_h = IMG_H // PATCH_SIZE

    # ------------------------------------------------------------- model
    classifier = build_model(
        args.model,
        in_channels=n_embedding,
        out_channels=NUM_CLASSES,
        token_w=token_w,
        token_h=token_h,
    )
    print(f"Loading weights from {args.model_path} …")
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded ✓")

    print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}")

    # -------------------------------------------------------- output dirs
    masks_dir = os.path.join(args.output_dir, "masks")
    masks_color_dir = os.path.join(args.output_dir, "masks_color")
    overlays_dir = os.path.join(args.output_dir, "overlays")
    comparisons_dir = os.path.join(args.output_dir, "comparisons")
    for d in [masks_dir, masks_color_dir, overlays_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ----------------------------------------------------------- inference
    print(f"\nProcessing {len(val_loader.dataset)} images …")

    all_ious, all_dices, all_paccs = [], [], []
    all_class_ious, all_class_dices = [], []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    sample_count = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for imgs, labels, data_ids in tqdm(val_loader, desc="Processing", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)

            tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = classifier(tokens)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)

            labels_sq = labels.squeeze(1).long()
            preds = outputs.argmax(dim=1)

            # Metrics
            iou, c_iou = compute_iou(outputs, labels_sq, return_per_class=True)
            dice, c_dice = compute_dice(outputs, labels_sq, return_per_class=True)
            pacc = compute_pixel_accuracy(outputs, labels_sq)
            all_ious.append(iou)
            all_dices.append(dice)
            all_paccs.append(pacc)
            all_class_ious.append(c_iou)
            all_class_dices.append(c_dice)

            # Per-image outputs
            for i in range(imgs.shape[0]):
                img_np = denormalize_image(imgs[i])
                img_uint8 = (img_np * 255).astype(np.uint8)
                gt_np = labels_sq[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()

                # Confusion matrix
                valid = gt_np < NUM_CLASSES
                np.add.at(confusion, (gt_np[valid].astype(int), pred_np[valid].astype(int)), 1)

                base = os.path.splitext(data_ids[i])[0]

                # Raw mask
                Image.fromarray(pred_np.astype(np.uint8)).save(
                    os.path.join(masks_dir, f"{base}_pred.png"))

                # Coloured mask
                pred_color = mask_to_color(pred_np.astype(np.uint8))
                cv2.imwrite(
                    os.path.join(masks_color_dir, f"{base}_pred_color.png"),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Overlay
                pred_resized = cv2.resize(
                    pred_color, (img_uint8.shape[1], img_uint8.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
                overlay = cv2.addWeighted(img_uint8, 0.5, pred_resized, 0.5, 0)
                cv2.imwrite(
                    os.path.join(overlays_dir, f"{base}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                # Comparison panels
                if sample_count < args.num_samples:
                    pred_tensor = torch.from_numpy(pred_np.astype(np.int64))
                    save_prediction_comparison(
                        imgs[i], labels_sq[i], pred_tensor,
                        os.path.join(comparisons_dir,
                                     f"sample_{sample_count:04d}_{base}.png"),
                        data_ids[i],
                    )
                sample_count += 1

    # ============================================================= #
    #  AGGREGATE METRICS                                             #
    # ============================================================= #
    mean_iou = float(np.nanmean(all_ious))
    mean_dice = float(np.nanmean(all_dices))
    mean_pacc = float(np.mean(all_paccs))
    avg_class_iou = np.nanmean(all_class_ious, axis=0).tolist()
    avg_class_dice = np.nanmean(all_class_dices, axis=0).tolist()

    # From confusion matrix
    cm_iou = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        cm_iou.append(float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan"))
    cm_mean_iou = float(np.nanmean(cm_iou))
    cm_acc = float(confusion.trace() / max(confusion.sum(), 1))

    # ============================================================= #
    #  PRINT RESULTS                                                 #
    # ============================================================= #
    print("\n" + "=" * 60)
    print(f"SEGMENTATION RESULTS — {NUM_CLASSES} classes")
    print("=" * 60)

    print(f"\n  Mean IoU       : {mean_iou:.4f}")
    print(f"  Mean Dice      : {mean_dice:.4f}")
    print(f"  Pixel Accuracy : {mean_pacc:.4f}")
    print(f"\n  Per-class IoU:")
    for idx in range(NUM_CLASSES):
        v = avg_class_iou[idx]
        bar = "█" * int(v * 40) if not np.isnan(v) else ""
        val_str = f"{v:.4f}" if not np.isnan(v) else "  N/A"
        print(f"    {CLASS_NAMES[idx]:<20} {val_str} {bar}")

    print(f"\n  Confusion-matrix IoU : {cm_mean_iou:.4f}")
    print(f"  Confusion-matrix Acc : {cm_acc:.4f}")
    print("=" * 60)

    # ---- Save results to text file ----
    results_path = os.path.join(args.output_dir, "metrics.txt")
    with open(results_path, "w") as f:
        f.write(f"SEGMENTATION METRICS — {NUM_CLASSES} classes\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean IoU         : {mean_iou:.4f}\n")
        f.write(f"Mean Dice        : {mean_dice:.4f}\n")
        f.write(f"Pixel Accuracy   : {mean_pacc:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for idx in range(NUM_CLASSES):
            f.write(f"  {CLASS_NAMES[idx]:<20} IoU={avg_class_iou[idx]:.4f}  Dice={avg_class_dice[idx]:.4f}\n")
        f.write(f"\nConfusion-matrix IoU : {cm_mean_iou:.4f}\n")
        f.write(f"Confusion-matrix Acc : {cm_acc:.4f}\n")
    print(f"\nMetrics saved to {results_path}")

    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  masks/          – raw class-id masks ({sample_count} images)")
    print(f"  masks_color/    – coloured RGB masks ({sample_count} images)")
    print(f"  overlays/       – image+prediction overlays ({sample_count} images)")
    print(f"  comparisons/    – side-by-side panels ({min(sample_count, args.num_samples)} images)")
    print("Inference complete! ✓")


if __name__ == "__main__":
    main()
