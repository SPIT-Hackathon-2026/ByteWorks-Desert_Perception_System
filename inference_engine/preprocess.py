"""
Image preprocessing for off-road segmentation (Inference version).
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter

def _box_filter(img, r):
    return uniform_filter(img.astype(np.float64), size=2*r+1, mode="reflect")

def _guided_filter(guide, p, r, eps):
    guide = guide.astype(np.float64)
    p = p.astype(np.float64)
    N = _box_filter(np.ones_like(guide), r)
    mean_I = _box_filter(guide, r) / N
    mean_p = _box_filter(p, r) / N
    mean_Ip = _box_filter(guide * p, r) / N
    mean_II = _box_filter(guide * guide, r) / N
    a = (mean_Ip - mean_I * mean_p) / (mean_II - mean_I * mean_I + eps)
    b = mean_p - a * mean_I
    return np.clip((_box_filter(a, r) / N) * guide + (_box_filter(b, r) / N), 0.01, 1.0)

def _rgb_to_gray(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def _rgb_to_hsv(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV) / np.array([360.0, 1.0, 1.0])

def _rgb_to_lab(img):
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[..., 0] = lab[..., 0] * 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab

def _lab_to_rgb(lab):
    lab_cv = lab.copy()
    lab_cv[..., 0] = np.clip(lab_cv[..., 0] * 255.0 / 100.0, 0, 255)
    lab_cv[..., 1] = np.clip(lab_cv[..., 1] + 128.0, 0, 255)
    lab_cv[..., 2] = np.clip(lab_cv[..., 2] + 128.0, 0, 255)
    return cv2.cvtColor(lab_cv.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0

def _dehaze_core(I):
    h, w = I.shape[:2]
    gray = _rgb_to_gray(I)
    avg_L, std_L = gray.mean(), gray.std()
    fog_idx = min(avg_L / (std_L + 0.05), 10.0)
    is_dense = fog_idx > 3.0

    n_top = max(round(h * w * 0.001), 20)
    A = np.clip([I[..., c].ravel()[np.argsort(I[..., c].ravel())[::-1][:n_top]].mean() * 0.96 for c in range(3)], 0.5, 0.98)
    
    omega = 0.55 + 0.35 * gaussian_filter(_rgb_to_gray(I), sigma=5)
    patch = 9 if is_dense else 7
    T = np.stack([1.0 - omega * cv2.erode((I[..., c]/(A[c]+1e-9)).astype(np.float32), np.ones((patch, patch))).astype(np.float64) for c in range(3)], axis=-1)
    
    guide = _rgb_to_gray(I)
    T_ref = np.clip(np.stack([_guided_filter(guide, T[..., c], 8, 7e-4) for c in range(3)], axis=-1), 0.05, 0.97)
    
    J = np.clip((I - A) / T_ref + A, 0, 1)
    lab = _rgb_to_lab(J)
    L = np.maximum(lab[..., 0] / 100.0, 1e-4)
    L_g = np.clip(L ** np.clip(np.log(0.5) / np.log(gaussian_filter(L, sigma=15) + 1e-9), 0.45, 1.8), 0, 1)
    lab[..., 0] = (L_g * (0.45 / (L_g.mean() + 1e-9))) * 100.0
    return _lab_to_rgb(lab)

def preprocess_image(img_uint8):
    img_f = img_uint8.astype(np.float64) / 255.0
    dehazed = _dehaze_core(img_f)
    lab = _rgb_to_lab(dehazed)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply((np.clip(lab[..., 0]/100.0, 0, 1)*255).astype(np.uint8)).astype(np.float64) / 255.0 * 100.0
    return (np.clip(_lab_to_rgb(lab), 0, 1) * 255).astype(np.uint8)
