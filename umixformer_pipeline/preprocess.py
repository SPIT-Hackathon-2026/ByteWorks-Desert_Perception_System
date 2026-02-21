"""
Image preprocessing for off-road segmentation.

Four preprocessing modes applied before model inference:
  1. **Defogging**   — atmospheric-light DCP-based haze removal
  2. **Sand-dust**   — colour-corrected dehazing for warm-tinted dust
  3. **Mist**        — gentle LAB-space haze removal
  4. **Histogram Equalisation** — CLAHE on luminance channel

All four are applied and combined adaptively based on the detected
degradation type, producing a single cleaned image per input.

Usage:
    from umixformer_pipeline.preprocess import preprocess_image, preprocess_batch
    cleaned = preprocess_image(rgb_uint8)      # (H, W, 3) uint8 → uint8
    batch   = preprocess_batch(list_of_uint8)  # list → list
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter


# ---------------------------------------------------------------------------
# Low-level helpers  (ported from imgpro.py)
# ---------------------------------------------------------------------------

def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    size = 2 * r + 1
    return uniform_filter(img.astype(np.float64), size=size, mode="reflect")


def _guided_filter(guide: np.ndarray, p: np.ndarray,
                   r: int, eps: float) -> np.ndarray:
    guide = guide.astype(np.float64)
    p = p.astype(np.float64)
    N = _box_filter(np.ones_like(guide), r)
    mean_I = _box_filter(guide, r) / N
    mean_p = _box_filter(p, r) / N
    mean_Ip = _box_filter(guide * p, r) / N
    mean_II = _box_filter(guide * guide, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, r) / N
    mean_b = _box_filter(b, r) / N
    return np.clip(mean_a * guide + mean_b, 0.01, 1.0)


def _bilateral_approx(img: np.ndarray, sigma_s: float, sigma_r: float,
                      iters: int = 4) -> np.ndarray:
    out = img.copy()
    for _ in range(iters):
        blurred = gaussian_filter(out, sigma=[sigma_s, sigma_s, 0])
        w = np.exp(-(out - blurred) ** 2 / (2 * sigma_r ** 2))
        out = w * out + (1 - w) * blurred
    return out


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV) / \
           np.array([360.0, 1.0, 1.0])


def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[..., 0] = lab[..., 0] * 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab_cv = lab.copy()
    lab_cv[..., 0] = np.clip(lab_cv[..., 0] * 255.0 / 100.0, 0, 255)
    lab_cv[..., 1] = np.clip(lab_cv[..., 1] + 128.0, 0, 255)
    lab_cv[..., 2] = np.clip(lab_cv[..., 2] + 128.0, 0, 255)
    return cv2.cvtColor(lab_cv.astype(np.uint8),
                        cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0


def _imerode_patch(img: np.ndarray, patch: int) -> np.ndarray:
    kernel = np.ones((patch, patch), dtype=np.uint8)
    return cv2.erode(img.astype(np.float32), kernel).astype(np.float64)


def _clahe_luminance(L: np.ndarray, clip: float = 0.01,
                     tiles: tuple = (8, 8)) -> np.ndarray:
    L8 = (np.clip(L, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip * 255, tileGridSize=tiles)
    return clahe.apply(L8).astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# Degradation classifier
# ---------------------------------------------------------------------------

def _classify_degradation(I: np.ndarray):
    """Returns (is_sandust, is_mist, is_dense, fog_idx)."""
    R, G, B = I[..., 0], I[..., 1], I[..., 2]
    avg_R, avg_G, avg_B = R.mean(), G.mean(), B.mean()

    hsv = _rgb_to_hsv(I)
    avg_S = hsv[..., 1].mean()
    gray = _rgb_to_gray(I)
    avg_L = gray.mean()
    std_L = gray.std()
    fog_idx = min(avg_L / (std_L + 0.05), 10.0)

    rg = avg_R / (avg_G + 1e-9)
    rb = avg_R / (avg_B + 1e-9)

    is_sandust = (rg > 1.04) and (rb > 1.15) and (avg_S < 0.30)
    is_mist = avg_B > avg_R + 0.02
    is_dense = fog_idx > 3.0
    return is_sandust, is_mist, is_dense, fog_idx


# ---------------------------------------------------------------------------
# Core dehazing pipeline  (condensed from imgpro.py)
# ---------------------------------------------------------------------------

def _dehaze_core(I: np.ndarray) -> np.ndarray:
    """Apply full dehazing pipeline. I: H×W×3 float64 [0,1]. Returns same."""
    h, w = I.shape[:2]
    is_sandust, is_mist, is_dense, _ = _classify_degradation(I)
    _t = lambda cond, t, f: t if cond else f  # noqa: E731

    # Bilateral pre-denoise
    I_f = _bilateral_approx(I, _t(is_sandust, 2.0, 1.5), 0.08)

    # Atmospheric light
    n_top = max(round(h * w * 0.001), 20)
    A = np.zeros(3)
    for c in range(3):
        flat = I_f[..., c].ravel()
        idx = np.argsort(flat)[::-1]
        A[c] = flat[idx[:n_top]].mean() * 0.96
    if is_sandust:
        A[0] = max(A[0], A[1])
        A[1] = max(A[1], A[2])
        A[2] = min(A[2], A[1] * 0.90)
    A = np.clip(A, 0.50, 0.98)

    # Texture map
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk
    gray_u8 = (_rgb_to_gray(I_f) * 255).astype(np.uint8)
    ent = rank_entropy(gray_u8, disk(4)).astype(np.float64)
    P_tex = (ent - ent.min()) / (ent.max() - ent.min() + 1e-9)
    P_tex = gaussian_filter(P_tex, sigma=5)

    # Transmission
    patch = _t(is_dense, 9, 7)
    om_min = _t(is_sandust, 0.60, 0.55)
    om_max = _t(is_sandust, _t(is_dense, 0.95, 0.88), 0.90)
    omega = om_min + (om_max - om_min) * P_tex

    T = np.zeros((h, w, 3))
    for c in range(3):
        norm_c = np.maximum(I_f[..., c] / (A[c] + 1e-9), 0.0)
        T[..., c] = 1.0 - omega * _imerode_patch(norm_c, patch)

    hsv = _rgb_to_hsv(I)
    avg_S = hsv[..., 1].mean()
    t_sat = 1.0 - (1.0 - avg_S) * np.maximum(omega - 0.1, 0.0)
    t_sat = gaussian_filter(t_sat, sigma=3)

    t_min = _t(is_sandust, 0.10, 0.05)
    if is_sandust:
        t_fused = T.copy()
    else:
        t_min_ch = T.min(axis=2)
        t_mono = 0.75 * t_min_ch + 0.25 * t_sat
        t_fused = np.stack([t_mono] * 3, axis=-1)

    for c in range(3):
        t_fused[..., c] = np.clip(t_fused[..., c], t_min, 0.97)

    # Guided filter refinement
    gf_r = max(round(min(h, w) * 0.03), 8)
    gf_eps = _t(is_sandust, 1.5e-3, 7e-4)
    guide = _rgb_to_gray(I_f)
    T_ref = np.zeros((h, w, 3))
    for c in range(3):
        T_ref[..., c] = np.clip(
            _guided_filter(guide, t_fused[..., c], gf_r, gf_eps), t_min, 0.97)

    # Scene recovery
    d_off = np.zeros(3)
    if is_sandust:
        avg_R = I_f[..., 0].mean()
        for c in range(3):
            d_off[c] = avg_R - I_f[..., c].mean()
        d_off = np.clip(d_off, -0.05, 0.20)
    A_eff = np.clip(A - d_off, 0.30, 0.98)

    J = np.zeros_like(I)
    for c in range(3):
        J[..., c] = (I[..., c] - A_eff[c]) / T_ref[..., c] + A_eff[c]
    J = np.clip(J, 0.0, 1.0)

    # Blue compensation (sand-dust)
    if is_sandust:
        ref_mean = (J[..., 0].mean() + J[..., 1].mean()) / 2.0
        deficit = max(ref_mean - J[..., 2].mean(), 0.0)
        alpha = min(deficit * 2.5, 0.45)
        J[..., 2] = np.clip(J[..., 2] + alpha * (J[..., 0] + J[..., 1]) / 2, 0, 1)
        lab = _rgb_to_lab(J)
        lab[..., 1] -= 0.4 * lab[..., 1].mean()
        lab[..., 2] -= 0.35 * lab[..., 2].mean()
        J = _lab_to_rgb(lab)

    if is_mist and not is_sandust:
        lab = _rgb_to_lab(J)
        lab[..., 1] = gaussian_filter(lab[..., 1], sigma=1.0)
        lab[..., 2] = gaussian_filter(lab[..., 2], sigma=1.0)
        J = _lab_to_rgb(lab)

    # Gamma contrast
    lab = _rgb_to_lab(J)
    L = np.maximum(lab[..., 0] / 100.0, 1e-4)
    L_loc = np.maximum(gaussian_filter(L, sigma=15), 0.01)
    gamma = np.clip(np.log(0.5) / np.log(L_loc + 1e-9), 0.45, 1.80)
    L_g = np.clip(L ** gamma, 0, 1)
    cur = L_g.mean()
    if cur > 0.01:
        L_g = np.clip(L_g * np.clip(0.45 / cur, 0.70, 1.40), 0, 1)
    lab[..., 0] = L_g * 100.0
    boost = _t(is_sandust, 1.25, 1.18)
    lab[..., 1] = np.clip(lab[..., 1] * boost, -128, 127)
    lab[..., 2] = np.clip(lab[..., 2] * boost, -128, 127)
    J = _lab_to_rgb(lab)

    # CLAHE
    clip_lim = _t(is_sandust, 0.008, 0.012)
    lab = _rgb_to_lab(J)
    lab[..., 0] = _clahe_luminance(lab[..., 0] / 100.0, clip_lim) * 100.0
    J = _lab_to_rgb(lab)

    # Light sharpen
    gray_J = _rgb_to_gray(J)
    gy, gx = np.gradient(gray_J)
    sharp = 0.3
    J_blur = np.stack([gaussian_filter(J[..., c], sigma=0.85) for c in range(3)], axis=-1)
    for c in range(3):
        J[..., c] = J[..., c] + sharp * (J[..., c] - J_blur[..., c])
    J = np.clip(J, 0, 1)

    # Final smooth
    J = np.stack([gaussian_filter(J[..., c], sigma=0.35) for c in range(3)], axis=-1)
    return np.clip(J, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Histogram equalisation (CLAHE on luminance)
# ---------------------------------------------------------------------------

def _histogram_equalize(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalisation. img: H×W×3 float64 [0,1]."""
    lab = _rgb_to_lab(img)
    L = lab[..., 0] / 100.0
    L8 = (np.clip(L, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L8_eq = clahe.apply(L8)
    lab[..., 0] = L8_eq.astype(np.float64) / 255.0 * 100.0
    return np.clip(_lab_to_rgb(lab), 0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(img_uint8: np.ndarray) -> np.ndarray:
    """Full preprocessing: dehaze + histogram equalize.

    Parameters
    ----------
    img_uint8 : (H, W, 3) uint8 RGB

    Returns
    -------
    cleaned : (H, W, 3) uint8 RGB
    """
    img_f = img_uint8.astype(np.float64) / 255.0

    # 1. Dehaze (handles fog / sand-dust / mist automatically)
    dehazed = _dehaze_core(img_f)

    # 2. Histogram equalisation on the dehazed result
    equalised = _histogram_equalize(dehazed)

    return (np.clip(equalised, 0, 1) * 255).astype(np.uint8)


def preprocess_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    """Apply preprocessing to a list of uint8 RGB images."""
    return [preprocess_image(img) for img in images]
