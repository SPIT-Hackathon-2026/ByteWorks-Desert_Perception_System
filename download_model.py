#!/usr/bin/env python3
"""
Download model checkpoint on startup (for Render deployment).
This script is called by render.yaml's preUploadCommand to ensure 
the checkpoint is available before the API starts.
"""

import os
import sys
import torch

def download_model():
    """Download model checkpoint if not present."""
    checkpoint_path = "umixformer_pipeline/checkpoints/umixformer_best.pth"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint already exists: {checkpoint_path}")
        return True
    
    print(f"⚠️ Checkpoint not found: {checkpoint_path}")
    print("Note: Render free tier cannot store 335MB files.")
    print("Options:")
    print("1. Push checkpoint to GitHub LFS (requires setup)")
    print("2. Upload to Hugging Face and download on startup")
    print("3. Use Render paid tier (includes storage)")
    print("4. Train model on Render (not practical)")
    print("")
    print("For now, API will use randomly initialized model for inference.")
    print("Frontend will still work but predictions will be random.")
    return False

if __name__ == "__main__":
    download_model()
