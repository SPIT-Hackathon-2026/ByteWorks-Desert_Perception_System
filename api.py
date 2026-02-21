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

from inference_engine.config import (
    CLASS_NAMES,
    COLOR_PALETTE,
    DEVICE,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)
from inference_engine.model import UMixFormerSeg
from inference_engine.utils import mask_to_color
from inference_engine.preprocess import preprocess_image as run_preprocess

# ─── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Desert Perception API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global singletons (loaded once) ────────────────────────────────────
_model = None
_device = DEVICE

def _get_model():
    """Lazy-load U-MixFormer model."""
    global _model

    if _model is None:
        print("Loading U-MixFormer model …")
        _model = UMixFormerSeg(pretrained_encoder=False).to(_device)
        
        # Load best checkpoint from umixformer_pipeline
        ckpt_path = "/home/raj_99/Projects/SPIT_Hackathon/umixformer_pipeline/checkpoints/umixformer_best.pth"
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded weights: {ckpt_path}")
        else:
            print(f"⚠  No checkpoint found at {ckpt_path}")

        _model.eval()

    return _model


# ─── Helpers ─────────────────────────────────────────────────────────────

def _preprocess_image(pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Resize + normalise a PIL image for U-MixFormer (384x384)."""
    pil_img = pil_img.convert("RGB")
    original_np = np.array(pil_img)

    resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
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
    return {"status": "ok", "model": "U-MixFormer", "device": str(_device)}


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    """Run segmentation on an uploaded image using U-MixFormer."""
    t0 = time.time()

    model = _get_model()

    # Read uploaded image
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw))
    input_tensor, original_np = _preprocess_image(pil_img)

    # ── NN inference ─────────────────────────────────────────────────
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=_device.type == "cuda"):
            logits = model(input_tensor)
            # Resize output to 384x384 if needed (model already interpolates)
            outputs = F.interpolate(
                logits,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

    final = outputs.argmax(dim=1)[0].cpu().numpy()
    t_inference = time.time() - t0

    # ── Build response visuals ───────────────────────────────────────
    color_mask = mask_to_color(final.astype(np.uint8))

    img_resized = cv2.resize(original_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img_resized, 0.5, color_mask, 0.5, 0)

    # Preprocessing (Defogged)
    img_preprocessed = run_preprocess(img_resized)

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
            "source": "U-MixFormer",
        })

    class_distribution.sort(key=lambda x: x["percentage"], reverse=True)

    # Terrain grid for 3D (downsampled)
    ds = 8
    small_h, small_w = IMG_SIZE // ds, IMG_SIZE // ds
    small_mask = cv2.resize(
        final.astype(np.uint8),
        (small_w, small_h),
        interpolation=cv2.INTER_NEAREST,
    )
    terrain_grid = small_mask.tolist()

    return JSONResponse({
        "original_b64": _ndarray_to_b64png(img_resized),
        "mask_b64": _ndarray_to_b64png(color_mask),
        "overlay_b64": _ndarray_to_b64png(overlay),
        "defog_b64": _ndarray_to_b64png(img_preprocessed),
        "class_distribution": class_distribution,
        "terrain_grid": terrain_grid,
        "inference_ms": round(t_inference * 1000, 1),
        "image_size": {"w": IMG_SIZE, "h": IMG_SIZE},
        "pipeline": {
            "backbone": "ConvNeXt (Hierarchical Features)",
            "head": "U-MixFormer Decoder (Mix-Attention)",
            "classes": CLASS_NAMES,
            "total_classes": NUM_CLASSES,
        },
    })


@app.get("/api/model-info")
def model_info():
    """Return model architecture details for the frontend."""
    return {
        "backbone": {
            "name": "ConvNeXt-Tiny",
            "source": "ImageNet-22k pretrained",
            "stages": 4,
            "channels": [96, 192, 384, 768],
        },
        "head": {
            "name": "U-MixFormer Decoder",
            "mechanism": "Mix-Attention",
            "fusion": "Multi-scale Progressive Refinement",
            "params": "4.1M",
        },
        "input": {
            "original": "960x540",
            "resized": f"{IMG_SIZE}x{IMG_SIZE}",
        },
        "classes": {
            "total": NUM_CLASSES,
            "names": CLASS_NAMES,
        },
    }

