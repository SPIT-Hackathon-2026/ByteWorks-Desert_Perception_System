"""
Gradient-based Explainability (XAI) for the U-MixFormer pipeline.

Uses per-class gradient saliency — processes one image at a time,
works within 6 GB VRAM.  Produces per-image:

  - **Waterfall plot** — per-class contribution bar chart
  - **Per-class heatmaps** — spatial gradient overlaid on the image
  - **Tiled overview** — all classes side by side
  - **Global summary bar chart** — mean importance across all images
  - **CSV table** — per-image class contributions + predicted probs

Usage (standalone)::

    python -m umixformer_pipeline.explain_shap
    python -m umixformer_pipeline.explain_shap --num_images 10

Or via evaluate.py::

    python -m umixformer_pipeline.evaluate --xai
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
from umixformer_pipeline.dataset import get_val_augmentations, OffroadSegDataset
from umixformer_pipeline.utils import denormalize_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Gradient saliency  (one image at a time — no extra VRAM)
# ============================================================================

def _compute_grad_attribution(model, img_tensor, device):
    """Compute per-class gradient saliency for a single image.

    Parameters
    ----------
    model      : nn.Module  (eval mode, on *device*)
    img_tensor : (3, H, W) float tensor (normalised)

    Returns
    -------
    per_class_maps       : (NUM_CLASSES, H, W) np.float64
    per_class_importance : (NUM_CLASSES,) np.float64
    pred_probs           : (NUM_CLASSES,) np.float64
    """
    inp = img_tensor.unsqueeze(0).to(device).requires_grad_(True)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(inp)  # (1, C, H, W)

    probs = F.softmax(logits, dim=1).mean(dim=(2, 3))  # (1, C)

    maps = np.zeros((NUM_CLASSES, inp.shape[2], inp.shape[3]), dtype=np.float64)
    importance = np.zeros(NUM_CLASSES, dtype=np.float64)

    for c in range(NUM_CLASSES):
        model.zero_grad()
        if inp.grad is not None:
            inp.grad.zero_()
        probs[0, c].backward(retain_graph=True)

        grad = inp.grad[0].detach().cpu().numpy()  # (3, H, W)
        spatial = np.abs(grad).mean(axis=0)          # (H, W)
        maps[c] = spatial
        importance[c] = spatial.mean()

    pred_probs = probs[0].detach().cpu().numpy()
    inp.requires_grad_(False)

    return maps, importance, pred_probs


# ============================================================================
# Plotting helpers
# ============================================================================

def _waterfall_plot(importance: np.ndarray, class_names: list[str],
                    predicted_class: int, save_path: str):
    """Horizontal bar chart of per-class gradient importance."""
    n = len(class_names)
    sorted_idx = np.argsort(importance)  # ascending

    vals = importance[sorted_idx]
    names = [class_names[i] for i in sorted_idx]
    colours = [tuple(c / 255 for c in COLOR_PALETTE[i]) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.8)))
    bars = ax.barh(range(n), vals, color=colours,
                   edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Mean |gradient|  (contribution to prediction)", fontsize=10)
    ax.set_title(f"Waterfall — Predicted: {class_names[predicted_class]}",
                 fontsize=12, fontweight="bold")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_bar_plot(mean_imp: np.ndarray, class_names: list[str],
                      save_path: str):
    """Global summary bar chart across all explained images."""
    n = len(class_names)
    sorted_idx = np.argsort(mean_imp)[::-1]

    colours = [tuple(c / 255 for c in COLOR_PALETTE[i]) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.8)))
    ax.barh(range(n), mean_imp[sorted_idx], color=colours,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([class_names[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel("Mean |gradient| importance", fontsize=10)
    ax.set_title("Global Summary — Mean Feature Importance per Class",
                 fontsize=12, fontweight="bold")

    for i, idx in enumerate(sorted_idx):
        ax.text(mean_imp[idx] + 0.0003, i, f"{mean_imp[idx]:.4f}",
                va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap_overlay(img_uint8: np.ndarray, spatial_map: np.ndarray) -> np.ndarray:
    """Overlay a JET heatmap on the original image."""
    abs_map = np.abs(spatial_map)
    mx = abs_map.max()
    norm = (abs_map / mx) if mx > 1e-8 else abs_map

    h, w = img_uint8.shape[:2]
    norm_resized = cv2.resize(norm.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap((norm_resized * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * img_uint8.astype(np.float32) +
               0.5 * heatmap.astype(np.float32)).astype(np.uint8)
    return overlay


def _tile_images(panels: list[np.ndarray], titles: list[str]) -> np.ndarray:
    """Horizontally tile same-sized RGB images with text labels."""
    h, w = panels[0].shape[:2]
    header = 30
    canvas = np.ones((h + header, w * len(panels), 3), dtype=np.uint8) * 255
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x0 = i * w
        canvas[header: header + h, x0: x0 + w] = panel
        cv2.putText(canvas, title, (x0 + 5, header - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# ============================================================================
# Main generator
# ============================================================================

def generate_shap_explanations(
    model: torch.nn.Module,
    dataset: OffroadSegDataset,
    output_dir: str,
    device: torch.device = DEVICE,
    num_images: int = 5,
    **_kwargs,
) -> None:
    """Generate gradient-saliency explanations for *num_images*.

    Saves per image:
      - ``{name}_xai_class{c}_{ClassName}.png``  — heatmap overlay
      - ``{name}_xai_waterfall.png``              — waterfall bar chart
      - ``{name}_xai_all.png``                    — tiled overview
    Also:
      - ``xai_global_summary.png``  — mean importance across all images
      - ``xai_analysis.csv``        — per-image table
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Free any leftover VRAM
    torch.cuda.empty_cache()

    n = min(num_images, len(dataset))
    print(f"\nGenerating gradient-saliency XAI for {n} images …")

    all_importance: list[np.ndarray] = []
    csv_rows: list[dict] = []

    for idx in tqdm(range(n), desc="XAI", unit="img"):
        sample = dataset[idx]
        if dataset.return_filename:
            img_tensor, _, fname = sample
            base_name = os.path.splitext(fname)[0]
        else:
            img_tensor, _ = sample
            base_name = f"img_{idx:04d}"

        img_uint8 = denormalize_image(img_tensor)  # (H, W, 3) uint8

        # --- Gradient attribution (single image, minimal VRAM) ---
        maps, importance, pred_probs = _compute_grad_attribution(
            model, img_tensor, device
        )
        pred_class = int(pred_probs.argmax())
        all_importance.append(importance)

        # --- Per-class heatmaps ---
        panels = []
        for c in range(NUM_CLASSES):
            overlay = _heatmap_overlay(img_uint8, maps[c])
            panels.append(overlay)
            out_path = os.path.join(
                output_dir,
                f"{base_name}_xai_class{c}_{CLASS_NAMES[c]}.png",
            )
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # --- Waterfall plot ---
        _waterfall_plot(
            importance, CLASS_NAMES, pred_class,
            os.path.join(output_dir, f"{base_name}_xai_waterfall.png"),
        )

        # --- Tiled overview ---
        tile = _tile_images(panels, CLASS_NAMES)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_xai_all.png"),
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
        )

        # CSV row
        row = {"image": base_name, "predicted": CLASS_NAMES[pred_class]}
        for c in range(NUM_CLASSES):
            row[f"importance_{CLASS_NAMES[c]}"] = f"{importance[c]:.6f}"
            row[f"prob_{CLASS_NAMES[c]}"] = f"{pred_probs[c]:.4f}"
        csv_rows.append(row)

        # Free grads
        torch.cuda.empty_cache()

    # --- Global summary ---
    if all_importance:
        mean_imp = np.mean(np.array(all_importance), axis=0)
        _summary_bar_plot(
            mean_imp, CLASS_NAMES,
            os.path.join(output_dir, "xai_global_summary.png"),
        )

        print(f"\n{'='*60}")
        print("  XAI Class-Level Importance (Mean |Gradient|)")
        print(f"{'='*60}")
        print(f"  {'Class':<14} {'Mean Importance':>16}")
        print(f"  {'-'*32}")
        for c in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[c]:<14} {mean_imp[c]:>16.6f}")
        print(f"{'='*60}")

    # Save CSV
    if csv_rows:
        csv_path = os.path.join(output_dir, "xai_analysis.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV → {csv_path}")

    print(f"XAI explanations saved → {output_dir}/")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate gradient-saliency XAI (U-MixFormer)")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(MODEL_SAVE_DIR, "umixformer_best.pth"))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PREDICTIONS_DIR, "shap_explanations"))
    p.add_argument("--num_images", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    from umixformer_pipeline.model import UMixFormerSeg
    model = UMixFormerSeg(pretrained_encoder=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    data_dir = TEST_DIR if args.split == "test" else VAL_DIR
    dataset = OffroadSegDataset(
        data_dir, augmentations=get_val_augmentations(), return_filename=True
    )
    print(f"Dataset: {len(dataset)} images ({args.split})")

    generate_shap_explanations(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()
