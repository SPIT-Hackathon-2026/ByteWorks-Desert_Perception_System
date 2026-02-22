# Deployment Guide

## Overview
This guide covers deploying the Desert Perception API to production using Vercel (frontend) + Render (backend).

## Architecture
```
┌─────────────────────────────────────────┐
│  Frontend (Vercel)                      │
│  - Next.js 15 + React                   │
│  - Tailwind CSS v4                      │
│  - semantic-segmentation-raj.vercel.app │
└──────────┬──────────────────────────────┘
           │ HTTPS API calls
           ↓
┌─────────────────────────────────────────┐
│  Backend (Render)                       │
│  - FastAPI + Uvicorn                    │
│  - U-MixFormer segmentation model       │
│  - semantic-segmentation-api.onrender.com
└─────────────────────────────────────────┘
```

## Frontend Deployment (Vercel) ✅
Status: **DEPLOYED** at https://semantic-segmentation-raj.vercel.app

### Setup
```bash
cd frontend
pnpm install
pnpm run build
vercel --prod  # Deploy to production
```

### Environment
- Framework: Next.js 15 (App Router)
- Styling: Tailwind CSS v4
- Deployment: Vercel (auto-scales with CDN)
- Domain: https://semantic-segmentation-raj.vercel.app

## Backend Deployment (Render) 🚀

### Known Issue: Model Checkpoint Storage
The 335MB model checkpoint cannot be deployed via Git on Render's free tier (Git LFS not supported). 

**Solutions:**

#### Option A: Local Testing (Dev/Demo)
Use the backend locally with the checkpoint already present:
```bash
cd /home/raj_99/Projects/SPIT_Hackathon
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
- Endpoint: http://localhost:8000
- Test: `curl http://localhost:8000/api/health`
- Upload: Post to http://localhost:8000/api/segment

#### Option B: Hugging Face Model Hosting (Recommended for Production)
1. Create Hugging Face account: https://huggingface.co
2. Create a private repo: `ByteWorks/semantic-segmentation`
3. Upload checkpoint via web UI or CLI:
   ```bash
   pip install huggingface-hub
   huggingface-cli login  # Paste your token
   huggingface-cli upload ByteWorks/semantic-segmentation \
     umixformer_pipeline/checkpoints/umixformer_best.pth
   ```
4. Update [api.py](api.py) with your HF repo name (line ~112):
   ```python
   repo_id="your-username/semantic-segmentation",
   ```
5. Commit & push to GitHub:
   ```bash
   git add api.py
   git commit -m "point to Hugging Face model repo"
   git push origin main
   ```
6. Render auto-deploys → first `/api/segment` request downloads model (~5-10s cold start)

#### Option C: Render Pro Tier ($7-12/month)
- Includes 100GB storage
- Push checkout directly in Git
- No cold starts for model loading
- https://render.com/pricing

### Current Status
- **Service**: Live at https://semantic-segmentation-api.onrender.com
- **Root**: `/` → 200 OK ✅
- **Health**: `/api/health` → Returns status (may be 503 if model missing)
- **Segment**: `/api/segment` → Requires model checkpoint (download on first request if HF configured)

### Debugging
Check Render logs:
```bash
# Via web: https://dashboard.render.com → Logs tab
# Look for:
# - "Model loaded" → Checkpoint found ✅
# - "Failed to load checkpoint" → Download from HF in progress
# - "CUDA out of memory" → May need GPU tier
```

### Testing from CLI
```bash
# Health check
curl https://semantic-segmentation-api.onrender.com/api/health

# Segment (requires local test image)
curl -X POST \
  -F "file=@test_image.jpg" \
  https://semantic-segmentation-api.onrender.com/api/segment \
  -o result.json
```

## Environment Variables
None required for basic operation. Optional for advanced configs:
- `HF_TOKEN` - For private HF repos (set in Render dashboard)
- `DEVICE` - Override device selection (auto-detects cuda/cpu)

## Frontend-Backend Integration
The frontend automatically routes API calls:
- Local dev: `http://localhost:8000`
- Production: `https://semantic-segmentation-api.onrender.com`

See [frontend/lib/api.ts](frontend/lib/api.ts) for API client configuration.

## Performance Characteristics
- **Image Processing**: 384×384 PNG → ~45ms (GPU), ~500ms (CPU)
- **Cold Start** (Render free):
  - Model download from HF: ~30s first time
  - Subsequent requests: <50ms
- **Typical Response Time**: 100-600ms (includes model load on cold start)

## Troubleshooting

### 502 Bad Gateway
1. Check Render logs for "Model loading failed"
2. Verify checkpoint exists (local) OR is in HF repo (cloud)
3. Try `/api/health` first to see detailed error

### CUDA Out of Memory
- Free tier has no GPU
- CPU inference works but slower (~500ms/image)
- Consider Render GPU tier or local testing

### Checkpoint Not Found
- Ensure `umixformer_best.pth` is in repo (if using Option C)
- OR configure HF download (Option B)
- OR test locally (Option A)

## Next Steps
1. ✅ Frontend deployed to Vercel
2. ⏳ Configure model hosting (HF or Render Pro)
3. ⏳ Full end-to-end testing
4. ⏳ Monitor Render logs for cold starts
5. ⏳ Optional: Add image caching for faster responses

## References
- [Render Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)
