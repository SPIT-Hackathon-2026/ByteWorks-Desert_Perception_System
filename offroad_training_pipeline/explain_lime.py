"""
LIME-based Explainability (XAI) for the segmentation pipeline.

Uses `lime.lime_image` to produce per-image explanations showing which
super-pixel regions contributed most to each predicted class.

Usage (standalone)::

    python -m offroad_training_pipeline.explain_lime \
        --model segformer_head \
        --data_dir dataset/Offroad_Segmentation_testImages \
        --num_images 10

Or call ``generate_lime_explanations()`` from your own code / test script.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from offroad_training_pipeline.config import (
    BACKBONE_SIZE,
    CLASS_NAMES,
    COLOR_PALETTE,
    DEVICE,
    IMG_H,
    IMG_W,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    PATCH_SIZE,
    PREDICTIONS_DIR,
    TEST_DIR,
)
from offroad_training_pipeline.dataset import get_image_transform, get_mask_transform, MaskDataset
from offroad_training_pipeline.models import build_model, load_backbone
from offroad_training_pipeline.models.backbone import get_embedding_dim
from offroad_training_pipeline.utils import denormalize_image


# ============================================================================
# Prediction wrapper for LIME
# ============================================================================

class _SegmentationPredictor:
    """Wraps backbone + head into a callable that LIME can use.

    LIME expects  ``f(batch_of_images) -> (N, C)``  probabilities, where *C*
    is the number of classes.  Since segmentation produces spatial maps we
    average the softmax probabilities over all pixels to obtain a single
    *image-level* class distribution – which is what LIME perturbs against.
    """

    def __init__(self, backbone, head, device: torch.device, img_h: int, img_w: int):
        self.backbone = backbone
        self.head = head
        self.device = device
        self.transform = get_image_transform(img_h, img_w)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : (N, H, W, 3)  uint8 RGB numpy array — LIME's convention.

        Returns
        -------
        probs : (N, NUM_CLASSES)  float64 array of per-image class probs.
        """
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            batch.append(self.transform(pil))

        tensor = torch.stack(batch).to(self.device)
        tokens = self.backbone.forward_features(tensor)["x_norm_patchtokens"]
        logits = self.head(tokens)
        logits = F.interpolate(logits, size=tensor.shape[2:],
                               mode="bilinear", align_corners=False)
        # softmax → spatial average → (N, C)
        probs = F.softmax(logits, dim=1).mean(dim=(2, 3))
        return probs.cpu().numpy()


# ============================================================================
# LIME explanation generator
# ============================================================================

def generate_lime_explanations(
    backbone,
    head,
    dataset: MaskDataset,
    output_dir: str,
    device: torch.device = DEVICE,
    num_images: int = 5,
    num_samples: int = 200,
    num_features: int = 15,
    top_labels: int | None = None,
) -> None:
    """Generate and save LIME explanations for *num_images* from *dataset*.

    For each image the function saves:

    * ``{name}_lime_class{c}.png``  – per-class heatmap overlay
    * ``{name}_lime_all.png``       – tiled overview of all classes

    Parameters
    ----------
    backbone, head : nn.Module
        Frozen backbone & trained segmentation head.
    dataset : MaskDataset
        Dataset to draw images from.
    output_dir : str
        Where to write explanation PNGs.
    num_images : int
        How many images to explain.
    num_samples : int
        LIME perturbation count (more = slower but sharper).
    num_features : int
        Max super-pixels to highlight.
    top_labels : int | None
        Explain this many top-predicted classes (*None* = all).
    """
    try:
        from lime import lime_image
    except ImportError:
        print("ERROR: `lime` package not installed. Run:  pip install lime")
        return

    os.makedirs(output_dir, exist_ok=True)
    predictor = _SegmentationPredictor(backbone, head, device, IMG_H, IMG_W)
    explainer = lime_image.LimeImageExplainer()

    if top_labels is None:
        top_labels = NUM_CLASSES

    n = min(num_images, len(dataset))
    print(f"\nGenerating LIME explanations for {n} images …")

    for idx in tqdm(range(n), desc="LIME", unit="img"):
        if dataset.return_filename:
            img_tensor, mask_tensor, fname = dataset[idx]
            base_name = os.path.splitext(fname)[0]
        else:
            img_tensor, mask_tensor = dataset[idx]
            base_name = f"img_{idx:04d}"

        # Convert to HWC uint8 RGB for LIME
        img_np = denormalize_image(img_tensor)
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Run explanation
        explanation = explainer.explain_instance(
            img_uint8,
            predictor,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            batch_size=8,
        )

        # --- per-class heatmaps ------------------------------------------
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
                # class was not among top_labels → skip
                temp = img_uint8.copy()
                mask = np.zeros(img_uint8.shape[:2], dtype=bool)

            # Tint the positive-attribution region with the class colour
            overlay = img_uint8.copy()
            colour = COLOR_PALETTE[class_id].tolist()
            overlay[mask.astype(bool)] = (
                0.5 * overlay[mask.astype(bool)] + 0.5 * np.array(colour)
            ).astype(np.uint8)

            # Save individual
            out_path = os.path.join(output_dir, f"{base_name}_lime_class{class_id}_{CLASS_NAMES[class_id]}.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            panels.append(overlay)

        # --- tiled overview (1 row) --------------------------------------
        tile = _tile_images(panels, CLASS_NAMES)
        tile_path = os.path.join(output_dir, f"{base_name}_lime_all.png")
        cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print(f"LIME explanations saved → {output_dir}/")


def _tile_images(panels: list[np.ndarray], titles: list[str]) -> np.ndarray:
    """Horizontally tile a list of same-sized RGB images with text labels."""
    h, w = panels[0].shape[:2]
    header = 30  # px for title text
    canvas_w = w * len(panels)
    canvas_h = h + header
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, (panel, title) in enumerate(zip(panels, titles)):
        x0 = i * w
        canvas[header: header + h, x0: x0 + w] = panel
        # Put title text (uses OpenCV)
        cv2.putText(canvas, title, (x0 + 5, header - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


# ============================================================================
# CLI entry-point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LIME explanations")
    parser.add_argument("--model", type=str, default="segformer_head")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backbone_size", type=str, default=BACKBONE_SIZE)
    parser.add_argument("--data_dir", type=str, default=TEST_DIR)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PREDICTIONS_DIR, "lime_explanations"))
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="LIME perturbation count (more ≈ sharper, slower)")
    parser.add_argument("--num_features", type=int, default=15)
    args = parser.parse_args()
    if args.model_path is None:
        best = os.path.join(MODEL_SAVE_DIR, f"{args.model}_best.pth")
        final = os.path.join(MODEL_SAVE_DIR, f"{args.model}.pth")
        args.model_path = best if os.path.exists(best) else final
    return args


def main():
    args = parse_args()
    device = DEVICE
    print(f"Using device: {device}")

    backbone = load_backbone(args.backbone_size, device)

    # Build dataset (with filenames)
    from offroad_training_pipeline.dataset import get_image_transform, get_mask_transform
    dataset = MaskDataset(
        data_dir=args.data_dir,
        transform=get_image_transform(),
        mask_transform=get_mask_transform(),
        return_filename=True,
    )
    print(f"Dataset: {len(dataset)} images from {args.data_dir}")

    # Discover embedding dim
    sample, _, _ = dataset[0]
    n_embedding = get_embedding_dim(backbone, sample.unsqueeze(0))

    token_w = IMG_W // PATCH_SIZE
    token_h = IMG_H // PATCH_SIZE

    head = build_model(
        args.model,
        in_channels=n_embedding,
        out_channels=NUM_CLASSES,
        token_w=token_w,
        token_h=token_h,
    )
    print(f"Loading weights: {args.model_path}")
    head.load_state_dict(torch.load(args.model_path, map_location=device))
    head = head.to(device)
    head.eval()

    generate_lime_explanations(
        backbone=backbone,
        head=head,
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        num_images=args.num_images,
        num_samples=args.num_samples,
        num_features=args.num_features,
    )


if __name__ == "__main__":
    main()
