"""
Evaluation & inference script for U-MixFormer segmentation.

Runs trained model on val/test set and produces:
  - Per-class IoU / Dice / accuracy metrics
  - Predicted masks (greyscale + colour)
  - Overlay visualisations
  - Metrics text file

Usage
-----
    python -m umixformer_pipeline.evaluate
    python -m umixformer_pipeline.evaluate --checkpoint path/to/best.pth --split test
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from umixformer_pipeline.config import (
    CLASS_NAMES,
    COLOR_PALETTE,
    DEVICE,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    PREDICTIONS_DIR,
    TEST_DIR,
    VAL_DIR,
)
from umixformer_pipeline.dataset import build_test_loader, build_val_loader
from umixformer_pipeline.metrics import (
    compute_confusion_matrix,
    iou_from_confusion,
    dice_from_confusion,
    pixel_accuracy_from_confusion,
)
from umixformer_pipeline.model import UMixFormerSeg
from umixformer_pipeline.utils import mask_to_color, denormalize_image


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate U-MixFormer segmentation")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(MODEL_SAVE_DIR, "umixformer_best.pth"),
                   help="Path to model checkpoint")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--output_dir", type=str, default=PREDICTIONS_DIR)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--save_predictions", action="store_true", default=True)
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UMixFormerSeg:
    """Load trained model from checkpoint."""
    model = UMixFormerSeg(pretrained_encoder=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if "best_val_iou" in ckpt:
        print(f"  Checkpoint best val IoU: {ckpt['best_val_iou']:.4f}")
    return model


def save_prediction_images(image_tensor, pred_mask, gt_mask, fname,
                           output_dir, has_gt=True):
    """Save prediction as colour mask, overlay, and optionally comparison."""
    masks_dir = os.path.join(output_dir, "masks")
    color_dir = os.path.join(output_dir, "masks_color")
    overlay_dir = os.path.join(output_dir, "overlays")
    compare_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Raw mask (class indices)
    mask_img = Image.fromarray(pred_mask.astype(np.uint8))
    mask_img.save(os.path.join(masks_dir, fname))

    # Colour mask
    color_mask = mask_to_color(pred_mask)
    color_img = Image.fromarray(color_mask)
    color_img.save(os.path.join(color_dir, fname))

    # Denormalize input image
    img_np = denormalize_image(image_tensor)

    # Overlay (50% blend)
    overlay = (img_np.astype(np.float32) * 0.5 +
               color_mask.astype(np.float32) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(overlay_dir, fname))

    # Side-by-side comparison with GT
    if has_gt and gt_mask is not None:
        os.makedirs(compare_dir, exist_ok=True)
        gt_color = mask_to_color(gt_mask)
        h, w = img_np.shape[:2]
        # [original | gt | prediction | overlay]
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
        canvas[:, :w] = img_np
        canvas[:, w:2*w] = gt_color
        canvas[:, 2*w:3*w] = color_mask
        canvas[:, 3*w:] = overlay
        Image.fromarray(canvas).save(os.path.join(compare_dir, fname))


@torch.no_grad()
def evaluate(model, dataloader, device, output_dir, save_preds=True,
             has_filenames=True, has_gt=True):
    """Run full evaluation."""
    cm_total = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for batch in tqdm(dataloader, desc="Evaluating"):
        if has_filenames:
            images, masks, fnames = batch
        else:
            images, masks = batch
            fnames = [None]

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(images)

        preds = logits.argmax(dim=1)  # (B, H, W)

        # Resize predictions to mask size if they differ
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(
                preds.unsqueeze(1).float(), size=masks.shape[-2:],
                mode="nearest"
            ).squeeze(1).long()

        # Accumulate confusion matrix
        cm = compute_confusion_matrix(preds, masks.to(device), NUM_CLASSES)
        cm_total += cm.cpu()

        # Save predictions
        if save_preds and has_filenames:
            for i in range(images.shape[0]):
                pred_np = preds[i].cpu().numpy()
                gt_np = masks[i].cpu().numpy()
                img_t = images[i].cpu()
                save_prediction_images(
                    img_t, pred_np, gt_np, fnames[i],
                    output_dir, has_gt=has_gt
                )

    # Compute metrics
    per_iou, miou = iou_from_confusion(cm_total)
    per_dice, mdice = dice_from_confusion(cm_total)
    pix_acc = pixel_accuracy_from_confusion(cm_total)

    return {
        "mean_iou": miou,
        "mean_dice": mdice,
        "pixel_accuracy": pix_acc,
        "per_class_iou": {CLASS_NAMES[i]: per_iou[i].item()
                          for i in range(NUM_CLASSES)},
        "per_class_dice": {CLASS_NAMES[i]: per_dice[i].item()
                           for i in range(NUM_CLASSES)},
        "confusion_matrix": cm_total.numpy(),
    }


def print_and_save_metrics(metrics: dict, output_dir: str, split: str):
    """Pretty-print metrics and save to file."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results — {split.upper()} set")
    print(f"{'='*60}")
    print(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice:      {metrics['mean_dice']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"\n  Per-class IoU:")
    for name, v in metrics["per_class_iou"].items():
        print(f"    {name:12s}: {v:.4f}")
    print(f"\n  Per-class Dice:")
    for name, v in metrics["per_class_dice"].items():
        print(f"    {name:12s}: {v:.4f}")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{split}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"U-MixFormer Evaluation — {split.upper()} set\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mean IoU:       {metrics['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:      {metrics['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for name, v in metrics["per_class_iou"].items():
            f.write(f"  {name:12s}: {v:.4f}\n")
        f.write("\nPer-class Dice:\n")
        for name, v in metrics["per_class_dice"].items():
            f.write(f"  {name:12s}: {v:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Predictions saved to {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, DEVICE)

    # Build dataloader
    if args.split == "val":
        dataloader = build_test_loader(VAL_DIR, batch_size=args.batch_size)
        has_gt = True
    else:
        dataloader = build_test_loader(TEST_DIR, batch_size=args.batch_size)
        has_gt = True  # test set also has GT masks

    print(f"Evaluating on {args.split} set: {len(dataloader.dataset)} images\n")

    metrics = evaluate(
        model, dataloader, DEVICE, args.output_dir,
        save_preds=args.save_predictions,
        has_filenames=True, has_gt=has_gt
    )

    print_and_save_metrics(metrics, args.output_dir, args.split)


if __name__ == "__main__":
    main()
