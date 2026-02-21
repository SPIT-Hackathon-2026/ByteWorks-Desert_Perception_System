"""
FastAPI backend for the off-road segmentation pipeline.

Serves a /api/segment endpoint that accepts an image upload and returns:
 - segmentation mask (coloured PNG, base64)
 - overlay (image + mask blend, base64)
 - class distribution
 - terrain grid for 3D visualisation

Start:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from offroad_training_pipeline.config import (
    BACKBONE_SIZE,
    CLASS_NAMES,
    COLOR_PALETTE,
    DEVICE,
    IMG_H,
    IMG_W,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    PATCH_SIZE,
)
from offroad_training_pipeline.models import build_model, load_backbone
from offroad_training_pipeline.models.backbone import get_embedding_dim
from offroad_training_pipeline.utils import mask_to_color

# ─── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Off-Road Segmentation API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global singletons (loaded once) ────────────────────────────────────
_backbone = None
_classifier = None
_device = DEVICE
_model_name = "segformer_head"


def _get_model():
    """Lazy-load backbone and head."""
    global _backbone, _classifier

    if _backbone is None:
        print("Loading DINOv2 backbone …")
        _backbone = load_backbone(BACKBONE_SIZE, _device)

    if _classifier is None:
        token_w = IMG_W // PATCH_SIZE
        token_h = IMG_H // PATCH_SIZE

        # Determine embedding dim
        dummy = torch.randn(1, 3, IMG_H, IMG_W).to(_device)
        n_emb = get_embedding_dim(_backbone, dummy)

        _classifier = build_model(
            _model_name,
            in_channels=n_emb,
            out_channels=NUM_CLASSES,
            token_w=token_w,
            token_h=token_h,
        )

        # Auto-detect best checkpoint
        best = os.path.join(MODEL_SAVE_DIR, f"{_model_name}_best.pth")
        final = os.path.join(MODEL_SAVE_DIR, f"{_model_name}.pth")
        ckpt = best if os.path.exists(best) else final

        if os.path.exists(ckpt):
            _classifier.load_state_dict(
                torch.load(ckpt, map_location=_device, weights_only=True)
            )
            print(f"Loaded weights: {ckpt}")
        else:
            print(f"⚠  No checkpoint found at {ckpt}")

        _classifier = _classifier.to(_device)
        _classifier.eval()

    return _backbone, _classifier


# ─── Helpers ─────────────────────────────────────────────────────────────

def _preprocess_image(pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Resize + normalise a PIL image for DINOv2."""
    pil_img = pil_img.convert("RGB")
    original_np = np.array(pil_img)

    resized = pil_img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(_device), original_np


def _ndarray_to_b64png(arr: np.ndarray) -> str:
    """Encode an RGB uint8 array to a base64 PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Routes ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model": _model_name, "device": str(_device)}


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    """Run segmentation on an uploaded image."""
    t0 = time.time()

    backbone, classifier = _get_model()

    # Read uploaded image
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw))
    input_tensor, original_np = _preprocess_image(pil_img)

    # ── NN inference ─────────────────────────────────────────────────
    with torch.no_grad():
        tokens = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        with torch.amp.autocast("cuda", enabled=_device.type == "cuda"):
            logits = classifier(tokens)
            outputs = F.interpolate(
                logits,
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

    final = outputs.argmax(dim=1)[0].cpu().numpy()
    t_inference = time.time() - t0

    # ── Build response visuals ───────────────────────────────────────
    color_mask = mask_to_color(final.astype(np.uint8))

    img_resized = cv2.resize(original_np, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img_resized, 0.5, color_mask, 0.5, 0)

    # Class distribution
    total_px = final.size
    class_distribution = []
    for cid in range(NUM_CLASSES):
        count = int((final == cid).sum())
        pct = round(count / total_px * 100, 2)
        class_distribution.append({
            "id": cid,
            "name": CLASS_NAMES[cid],
            "percentage": pct,
            "color": f"rgb({COLOR_PALETTE[cid][0]},{COLOR_PALETTE[cid][1]},{COLOR_PALETTE[cid][2]})",
            "source": "NN",
        })

    class_distribution.sort(key=lambda x: x["percentage"], reverse=True)

    # Terrain grid for 3D (downsampled)
    ds = 8
    small_h, small_w = IMG_H // ds, IMG_W // ds
    small_mask = cv2.resize(
        final.astype(np.uint8),
        (small_w, small_h),
        interpolation=cv2.INTER_NEAREST,
    )
    terrain_grid = small_mask.tolist()

    return JSONResponse({
        "mask_b64": _ndarray_to_b64png(color_mask),
        "overlay_b64": _ndarray_to_b64png(overlay),
        "class_distribution": class_distribution,
        "terrain_grid": terrain_grid,
        "inference_ms": round(t_inference * 1000, 1),
        "image_size": {"w": IMG_W, "h": IMG_H},
        "pipeline": {
            "backbone": "DINOv2 ViT-S/14",
            "head": "SegFormer (4.4M params)",
            "classes": CLASS_NAMES,
            "total_classes": NUM_CLASSES,
        },
    })


@app.get("/api/model-info")
def model_info():
    """Return model architecture details for the frontend."""
    return {
        "backbone": {
            "name": "DINOv2 ViT-S/14",
            "source": "facebookresearch/dinov2",
            "embedding_dim": 384,
            "patch_size": PATCH_SIZE,
            "frozen": True,
        },
        "head": {
            "name": "SegFormer",
            "blocks": 4,
            "hidden_dim": 256,
            "heads": 8,
            "params": "4.4M",
        },
        "training": {
            "optimizer": "AdamW (lr=1e-3, wd=1e-2)",
            "scheduler": "OneCycleLR (cosine, 10% warmup)",
            "loss": "CrossEntropyLoss (class-weighted)",
            "amp": True,
            "epochs": 15,
            "batch_size": 2,
        },
        "classes": {
            "total": NUM_CLASSES,
            "names": CLASS_NAMES,
        },
        "input": {
            "original": "960x540",
            "resized": f"{IMG_W}x{IMG_H}",
        },
        "dataset": {
            "train": 2857,
            "val": 317,
            "test": 1002,
        },
    }
