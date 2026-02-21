"""
LIME-based Explainability (XAI) for the U-MixFormer pipeline.

Generates per-image, per-class heatmaps showing which super-pixel regions
contributed most to each predicted class.

Usage (standalone)::

    python -m umixformer_pipeline.explain_lime
    python -m umixformer_pipeline.explain_lime --num_images 10 --num_samples 300

Or call ``generate_lime_explanations()`` from evaluate.py via --lime flag.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from umixformer_pipeline.config import (
    CLASS_NAMES,
    COLOR_PALETTE,
    DEVICE,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    PREDICTIONS_DIR,
    TEST_DIR,
    VAL_DIR,
)
from umixformer_pipeline.dataset import get_val_augmentations, OffroadSegDataset
from umixformer_pipeline.utils import denormalize_image


# ============================================================================
# Prediction wrapper for LIME
# ============================================================================

class _SegmentationPredictor:
    """Wraps the U-MixFormer model into a LIME-compatible callable.

    LIME expects ``f(batch_of_images) -> (N, C)`` probabilities.
    We average the per-pixel softmax across all spatial locations to get a
    single image-level class distribution.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 img_size: int = IMG_SIZE):
        self.model = model
        self.device = device
        self.img_size = img_size
        # Build albumentations val transform (resize + normalise + to-tensor)
        self.aug = get_val_augmentations(img_size)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : (N, H, W, 3) uint8 RGB — LIME convention.

        Returns
        -------
        probs : (N, NUM_CLASSES) float64 — per-image class probabilities.
        """
        batch = []
        for img in images:
            transformed = self.aug(image=img.astype(np.uint8))
            batch.append(transformed["image"])  # (C, H, W) tensor

        tensor = torch.stack(batch).to(self.device)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(tensor)  # (N, C, H, W)

        # softmax → spatial average → (N, C)
        probs = F.softmax(logits, dim=1).mean(dim=(2, 3))
        return probs.cpu().numpy()


# ============================================================================
# LIME explanation generator
# ============================================================================

def generate_lime_explanations(
    model: torch.nn.Module,
    dataset: OffroadSegDataset,
    output_dir: str,
    device: torch.device = DEVICE,
    num_images: int = 5,
    num_samples: int = 200,
    num_features: int = 15,
    top_labels: int | None = None,
) -> None:
    """Generate and save LIME explanations for *num_images* from *dataset*.

    Saves per image:
      - ``{name}_lime_class{c}_{ClassName}.png`` — per-class overlay
      - ``{name}_lime_all.png`` — tiled overview of all classes
    """
    try:
        from lime import lime_image
    except ImportError:
        print("ERROR: `lime` package not installed.  Run:  uv pip install lime")
        return

    os.makedirs(output_dir, exist_ok=True)
    predictor = _SegmentationPredictor(model, device)
    explainer = lime_image.LimeImageExplainer()

    if top_labels is None:
        top_labels = NUM_CLASSES

    n = min(num_images, len(dataset))
    print(f"\nGenerating LIME explanations for {n} images …")

    for idx in tqdm(range(n), desc="LIME", unit="img"):
        sample = dataset[idx]
        if dataset.return_filename:
            img_tensor, _, fname = sample
            base_name = os.path.splitext(fname)[0]
        else:
            img_tensor, _ = sample
            base_name = f"img_{idx:04d}"

        # Convert normalised tensor → HWC uint8 RGB for LIME
        img_uint8 = denormalize_image(img_tensor)  # (H, W, 3) uint8

        explanation = explainer.explain_instance(
            img_uint8,
            predictor,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            batch_size=8,
        )

        # Per-class heatmaps
        panels = []
        for class_id in range(NUM_CLASSES):
            try:
                temp, mask = explanation.get_image_and_mask(
                    class_id,
                    positive_only=True,
                    num_features=num_features,
                    hide_rest=False,
                )
            except KeyError:
                temp = img_uint8.copy()
                mask = np.zeros(img_uint8.shape[:2], dtype=bool)

            overlay = img_uint8.copy()
            colour = COLOR_PALETTE[class_id].tolist()
            region = mask.astype(bool)
            overlay[region] = (
                0.5 * overlay[region].astype(np.float32)
                + 0.5 * np.array(colour, dtype=np.float32)
            ).astype(np.uint8)

            out_path = os.path.join(
                output_dir,
                f"{base_name}_lime_class{class_id}_{CLASS_NAMES[class_id]}.png",
            )
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            panels.append(overlay)

        # Tiled overview
        tile = _tile_images(panels, CLASS_NAMES)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_lime_all.png"),
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
        )

    print(f"LIME explanations saved → {output_dir}/")


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
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate LIME explanations (U-MixFormer)")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(MODEL_SAVE_DIR, "umixformer_best.pth"))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PREDICTIONS_DIR, "lime_explanations"))
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=200,
                   help="LIME perturbation count (more = sharper, slower)")
    p.add_argument("--num_features", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    # Load model
    from umixformer_pipeline.model import UMixFormerSeg
    model = UMixFormerSeg(pretrained_encoder=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Dataset
    data_dir = TEST_DIR if args.split == "test" else VAL_DIR
    dataset = OffroadSegDataset(
        data_dir, augmentations=get_val_augmentations(), return_filename=True
    )
    print(f"Dataset: {len(dataset)} images ({args.split})")

    generate_lime_explanations(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        num_images=args.num_images,
        num_samples=args.num_samples,
        num_features=args.num_features,
    )


if __name__ == "__main__":
    main()
