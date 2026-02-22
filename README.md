# 🏜️ Desert Perception System — ByteWorks | SPIT Hackathon 2026

> **End-to-end autonomous perception for desert/off-road terrain** — Real-time semantic segmentation using U-MixFormer, hardware sensor fusion, and a full-stack cloud-deployed interface.

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-000?style=for-the-badge&logo=vercel)](https://semantic-segmentation-raj.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://semantic-segmentation-api.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Hackathon-orange?style=for-the-badge)](./LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-system-architecture)
- [Features](#-key-features)
- [Terrain Classes](#-terrain-classes)
- [Hardware Integration](#-hardware-integration)
- [Model Performance](#-model-performance)
- [Robustness Testing](#-robustness-testing)
- [API Reference](#-api-reference)
- [Frontend](#-frontend)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Contributors](#-contributors)

---

## 🎯 Overview

The **Desert Perception System** is a multi-modal autonomous perception pipeline designed for Unmanned Ground Vehicles (UGVs) operating in sandy, arid, and off-road environments. The system fuses visual deep learning (semantic segmentation) with embedded hardware sensors (IR and Ultrasonic) to produce real-time terrain awareness and obstacle risk scores.

**Why this matters:** Conventional object detection models trained on urban datasets fail catastrophically in desert environments — sand dunes look like roads, rock formations occlude obstacles, and harsh lighting conditions destroy color cues. This system addresses those challenges directly with a domain-specific model, multi-spectral imaging, and a purpose-built sensor fusion layer.

### Live Demo

| Component | URL |
|---|---|
| 🌐 Web Frontend | [semantic-segmentation-raj.vercel.app](https://semantic-segmentation-raj.vercel.app) |
| ⚙️ REST API | [semantic-segmentation-api.onrender.com](https://semantic-segmentation-api.onrender.com) |
| 📖 API Docs | [semantic-segmentation-api.onrender.com/docs](https://semantic-segmentation-api.onrender.com/docs) |

---

## 🏗️ System Architecture

The system is composed of four integrated layers:

```
┌─────────────────────────────────────────────────────────┐
│                     SENSOR LAYER                        │
│   RGB Camera  │  IR Sensor  │  Ultrasonic  │  UV Cam   │
└───────┬───────────────┬──────────────┬──────────────────┘
        │               │              │
        ▼               ▼              ▼
┌───────────────┐ ┌──────────────────────────┐
│  VISION PATH  │ │     HARDWARE PATH        │
│ Preprocessing │ │  IR/Ultrasonic Ensemble  │
│   384×384     │ │  Obstacle Risk Score     │
│   Normalize   │ │  Proximity Alerts        │
└───────┬───────┘ └──────────┬───────────────┘
        │                    │
        ▼                    │
┌───────────────────┐        │
│   U-MixFormer     │        │
│  ConvNeXt Backbone│        │
│  Mix-Attention    │        │
│  Decoder (4.1M)   │        │
└───────┬───────────┘        │
        │                    │
        ▼                    ▼
┌──────────────────────────────────────┐
│         FUSION & RISK LAYER          │
│  7-Class Segmentation Mask           │
│  Obstacle Density Score              │
│  Terrain Complexity Index            │
│  Visibility Score                    │
│  Overall Risk Level (LOW/MED/HIGH)   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│           DEPLOYMENT LAYER           │
│   FastAPI Backend  ←→  Next.js SPA  │
│   Render (GPU)         Vercel (CDN) │
└──────────────────────────────────────┘
```

### Model Architecture: U-MixFormer

```
Input (384×384×3)
     │
     ▼
ConvNeXt-Tiny Backbone
     ├── Stage 1 → 96 channels  (96×96)
     ├── Stage 2 → 192 channels (48×48)
     ├── Stage 3 → 384 channels (24×24)
     └── Stage 4 → 768 channels (12×12)
                │
                ▼
     U-MixFormer Decoder
         ├── Multi-scale Feature Fusion
         ├── Mix-Attention Blocks (local + global)
         └── Progressive Upsampling Refinement
                │
                ▼
     Output (384×384×7)  →  7-Class Softmax Logits
```

---

## ✨ Key Features

- **U-MixFormer Segmentation** — 4.1M parameter decoder head on ConvNeXt-Tiny backbone (~32M total); ~45ms inference on RTX 3090
- **7-Class Terrain Segmentation** — Pixel-level classification of desert terrain into actionable categories
- **IR/Ultrasonic Sensor Fusion** — Hardware ensemble for proximity-based obstacle detection, independent of camera visibility
- **UV & IR Script Processing** — Multi-spectral analysis scripts for enhanced desert scene understanding under harsh lighting
- **Offroad-specific Training Pipeline** — Domain-adapted training with the Offroad Segmentation dataset + data augmentation
- **Weather Degradation Robustness** — Validated against synthetic FOG (intensity 0.70) and MIST (intensity 0.62) conditions
- **Real-time Risk Assessment** — Composite risk score (obstacle density + terrain complexity + visibility) → LOW / MEDIUM / HIGH
- **3D Pipeline Visualization** — Interactive Three.js architecture diagram with particle flow animation (`segheads.mp4`)
- **LIME Explainability** — Model transparency panel showing per-region feature attribution
- **Full Cloud Deployment** — Vercel (frontend) + Render (GPU backend) with auto-scaling and CI/CD via GitHub push

---

## 🗺️ Terrain Classes

| ID | Class | Color | Description |
|---|---|---|---|
| 0 | **Sky** | `#87CEFA` | Open sky above horizon |
| 1 | **Driveable** | `#90EE90` | Safe traversable sand / path |
| 2 | **Rock** | `#808080` | Solid rock formations |
| 3 | **Obstacle** | `#FF4444` | Dynamic or unknown obstacle |
| 4 | **Grass** | `#228B22` | Sparse desert vegetation |
| 5 | **Sand** | `#F4A460` | Loose sand — caution zone |
| 6 | **Rough** | `#8B4513` | Uneven, difficult terrain |

---

## 🔧 Hardware Integration

The `Hardware Code/` directory and `IR_UV_Scripts/` contain embedded firmware and processing scripts for the physical UGV sensor suite.

**Sensor Stack:**
- **Ultrasonic Sensor** — Distance-based obstacle detection, proximity alerts, range: 2cm–400cm
- **IR Sensor** — Passive infrared obstacle presence, works in complete darkness and dust
- **UV Camera** — Multi-spectral capture for improved sand/rock discrimination
- **IR Camera** — Thermal imaging for obstacle detection in fog and dust storms

**IR/Ultrasonic Ensemble Models** (`IR_Ultrasonic Models/`) combine both sensor outputs with a lightweight fusion model to produce a hardware-level risk score that is fused with the vision pipeline's output in the final risk assessment layer.

**Image Processing Algorithms** (`Image Processing Algs/`) include classical CV preprocessing for desert-specific challenges: sand glare normalization, horizon detection, and dust haze removal prior to DL inference.

---

## 📊 Model Performance

### Inference Speed

| Device | Resolution | FPS | Latency |
|---|---|---|---|
| A100 GPU | 384×384 | 25+ | ~40ms |
| RTX 3090 | 384×384 | ~22 | ~45ms |
| CPU (i9) | 384×384 | ~2 | ~500ms |

### Model Size

| Component | Parameters | Weight Size |
|---|---|---|
| ConvNeXt-Tiny Backbone | 28M | 105MB |
| U-MixFormer Decoder Head | 4.1M | 15.6MB |
| **Total** | **~32M** | **~120MB** |

Checkpoint: `umixformer_pipeline/checkpoints/umixformer_best.pth`

---

## 🌫️ Robustness Testing

The system was validated under two synthetic weather degradation conditions applied to 50 real test images each.

| Variant | Intensity | Visual Effect | Test Images |
|---|---|---|---|
| **FOG** | 0.70 | Dense grey-white uniform veil | 50 |
| **MIST** | 0.62 | Blue-tinted soft haze (Rayleigh scattering) | 50 |

**Results Summary:**

```
FOG Degradation:
  Avg Inference:     45.2ms  |  Throughput: ~22 img/sec
  Class Distribution: Sky 8.3% | Driveable 42.1% | Obstacle 28.4% | Rock 10.2%

MIST Degradation:
  Avg Inference:     44.8ms  |  Throughput: ~22 img/sec
  Class Distribution: Sky 12.1% | Driveable 45.3% | Obstacle 25.2% | Rock 9.4%

Model Stability: ✅ Consistent across all degradation variants
```

**Output artifacts** (in `dataset/results_better/`):

```
results_better/
├── robustness_metrics.json
├── robustness_metrics.txt
├── predictions_fog/
│   ├── input_images/     # Degraded FOG inputs
│   ├── masks/            # Raw segmentation masks
│   ├── masks_color/      # Color-coded class maps
│   ├── overlays/         # Input + mask blended
│   └── comparisons/      # [Original | GT | Pred | Overlay] side-by-side
└── predictions_mist/
    └── (same structure)
```

Run tests yourself:

```bash
uv run python test_robustness.py
```

---

## 📡 API Reference

Base URL: `https://semantic-segmentation-api.onrender.com`

### `GET /api/health`
Health check. Returns model status and compute device.

```json
{ "status": "ok", "model": "U-MixFormer", "device": "cuda" }
```

### `POST /api/segment`
Upload an image for segmentation. `multipart/form-data` with field `file` (PNG or JPEG).

**Response:**

```json
{
  "original_b64": "<base64>",
  "mask_b64": "<base64>",
  "overlay_b64": "<base64>",
  "defog_b64": "<base64>",
  "class_distribution": [
    { "id": 1, "name": "Driveable", "percentage": 42.1, "color": "rgb(144, 238, 144)" }
  ],
  "inference_ms": 45.3,
  "risk_assessment": {
    "risk_score": 0.4521,
    "risk_level": "MEDIUM",
    "obstacle_density": 0.5234,
    "uncertainty": 0.3891,
    "terrain_complexity": 0.2145,
    "visibility": 0.7823
  }
}
```

### `GET /api/model-info`
Returns model architecture details and configuration.

---

## 🎨 Frontend

Built with **Next.js 14** (TypeScript), deployed on Vercel.

**Key UI Components:**

| Component | Description |
|---|---|
| `hero-section.tsx` | Landing hero with animated entry |
| `upload-section.tsx` | Drag-and-drop image upload |
| `processing-pipeline.tsx` | Sequential animated reveal of pipeline stages |
| `output-dashboard.tsx` | Final segmentation results and class overlay |
| `statistics-panel.tsx` | Real-time metrics, pie chart, class breakdown |
| `terrain-3d.tsx` | Three.js 3D architecture visualization + `segheads.mp4` |
| `risk-gauge.tsx` | Animated risk level gauge (LOW/MEDIUM/HIGH) |
| `model-transparency.tsx` | LIME-based feature attribution explainability panel |

The 3D pipeline animation (`public/segheads.mp4`, 305MB) shows 7 segmentation heads with parallel branch processing and particle data-flow — rendered with Python + Matplotlib + FFmpeg via `animation.py`.

---

## 🗂️ Project Structure

```
ByteWorks-Desert_Perception_System/
│
├── api.py                          # FastAPI application entry point
├── main.py                         # CLI entry point / local testing
├── animation.py                    # 3D pipeline animation renderer
├── train_segment.py                # Training script
├── test_robustness.py              # Robustness evaluation pipeline
├── download_model.py               # Model weight downloader
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata (uv)
├── render.yaml                     # Render.com deployment config
│
├── umixformer_pipeline/            # Core model code
│   ├── model.py                    # U-MixFormer architecture
│   ├── config.py                   # Model + training config
│   ├── evaluate.py                 # Evaluation loop
│   ├── metrics.py                  # mIoU, pixel accuracy, etc.
│   └── checkpoints/
│       └── umixformer_best.pth     # Best model weights
│
├── inference_engine/               # Optimized inference wrapper
│   ├── model.py                    # Inference-only model wrapper
│   ├── config.py                   # Inference config
│   └── utils.py                    # Pre/post-processing utilities
│
├── offroad_training_pipeline/      # Domain-specific training pipeline
├── Offroad_Segmentation_Scripts/   # Dataset preprocessing scripts
│
├── Hardware Code/                  # Embedded firmware (UGV sensors)
├── IR_UV_Scripts/                  # IR/UV camera processing scripts
├── IR_Ultrasonic Models/           # Sensor fusion models
├── Image Processing Algs/          # Classical CV preprocessing
├── scripts/                        # Utility and helper scripts
│
├── frontend/                       # Next.js web application
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── hero-section.tsx
│   │   ├── upload-section.tsx
│   │   ├── processing-pipeline.tsx
│   │   ├── output-dashboard.tsx
│   │   ├── statistics-panel.tsx
│   │   ├── terrain-3d.tsx
│   │   ├── risk-gauge.tsx
│   │   ├── model-transparency.tsx
│   │   └── ui/
│   ├── hooks/
│   │   ├── use-mobile.ts
│   │   └── use-toast.ts
│   ├── public/
│   │   └── segheads.mp4            # 3D animation (305MB)
│   └── package.json
│
├── DEPLOYMENT_GUIDE.md
├── QUICKSTART.md
├── PROJECT_STRUCTURE.txt
└── results.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

```
Python 3.11+
CUDA 12.1+ (recommended for GPU inference)
Node.js 18+
pnpm (or npm/yarn)
uv (Python package manager)
```

### Backend

```bash
# Clone the repo
git clone https://github.com/SPIT-Hackathon-2026/ByteWorks-Desert_Perception_System.git
cd ByteWorks-Desert_Perception_System

# Create and activate Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install uv
uv pip install -r requirements.txt

# Download model weights
uv run python download_model.py

# Start the API server
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev
# Open http://localhost:3000
```

### Running Robustness Tests

```bash
# From project root
uv run python test_robustness.py
# Outputs saved to dataset/results_better/
```

---

## 🚀 Deployment

### Frontend (Vercel)

```bash
cd frontend
vercel --prod
```

### Backend (Render)

The `render.yaml` at the repo root configures auto-deployment. Simply push to `main`:

```bash
git add .
git commit -m "Deploy: <description>"
git push origin main
# Render detects the push and auto-redeploys
```

Verify the deployment:

```bash
curl https://semantic-segmentation-api.onrender.com/api/health
# {"status":"ok","model":"U-MixFormer","device":"cuda"}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Deep Learning | PyTorch 2.10 | Model training & inference |
| Architecture | U-MixFormer + ConvNeXt | Semantic segmentation backbone + decoder |
| API | FastAPI + Uvicorn | High-performance async REST API |
| Frontend | Next.js 14 + React | Web interface |
| 3D Rendering | Three.js + React Three Fiber | Architecture visualization |
| Animations | Framer Motion | UI transitions |
| Styling | Tailwind CSS v4 | Utility-first CSS |
| Export | Matplotlib + FFmpeg | 3D animation video generation |
| Hardware | Arduino / Embedded C | UGV sensor firmware |
| Sensor Processing | MATLAB | IR/UV image analysis |
| Frontend Host | Vercel | CDN + auto-scaling |
| Backend Host | Render | GPU cloud server |
| Package Manager | uv | Fast Python dependency management |

---

## 👥 Contributors

| Name | GitHub |
|---|---|
| Raj | [@CodeCraftsmanRaj](https://github.com/CodeCraftsmanRaj) |
| Shivani Bhat | [@shivanibhat24](https://github.com/shivanibhat24) |

---

## 📄 License

This project was developed for **SPIT Hackathon 2026** by Team ByteWorks. All rights reserved by the contributors.

---

*Last updated: February 22, 2026 · Status: ✅ Production Ready*
