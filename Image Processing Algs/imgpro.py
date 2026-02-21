"""
=========================================================================
 ULTRA DEHAZE v5 — SAND-DUST / FOG / MIST / HAZE SPECIALIST
 =========================================================================
 Python port of the original MATLAB implementation.
 Retains all physical models, algorithms, and pipeline steps.

 DEPENDENCIES:
   pip install numpy opencv-python-headless scipy scikit-image pillow matplotlib

 USAGE:
   python ultra_dehaze_v5.py [image_path]
   (or drop image in working directory and run without arguments)

 OUTPUT:
   defogging_output/<name>_dehazed.png
   defogging_output/<name>_comparison.png
=========================================================================
"""

import sys
import os
import glob
import warnings
import math

import numpy as np
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.measure import label
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# HELPER / FUNCTION LIBRARY
# ---------------------------------------------------------------------------

def ternary(cond, t, f):
    return t if cond else f


def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Fast box (mean) filter using uniform_filter."""
    size = 2 * r + 1
    return uniform_filter(img.astype(np.float64), size=size, mode='reflect')


def guided_filter_fast(guide: np.ndarray, p: np.ndarray,
                        r: int, eps: float) -> np.ndarray:
    """
    Fast guided filter (He et al. ECCV 2010).
    guide, p : H×W float64 in [0,1]
    """
    guide = guide.astype(np.float64)
    p = p.astype(np.float64)

    N = box_filter(np.ones_like(guide), r)
    mean_I = box_filter(guide, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(guide * p, r) / N
    mean_II = box_filter(guide * guide, r) / N

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r) / N
    mean_b = box_filter(b, r) / N

    out = mean_a * guide + mean_b
    return np.clip(out, 0.01, 1.0)


def bilateral_approx(img: np.ndarray, sigma_s: float, sigma_r: float,
                     iterations: int = 4) -> np.ndarray:
    """
    Approximated bilateral filter via iterative range-weighted Gaussian.
    img : H×W×3 float64 [0,1]
    """
    out = img.copy()
    for _ in range(iterations):
        blurred = gaussian_filter(out, sigma=[sigma_s, sigma_s, 0])
        w = np.exp(-(out - blurred) ** 2 / (2 * sigma_r ** 2))
        out = w * out + (1 - w) * blurred
    return out


def structure_tensor(img: np.ndarray, sigma_i: float, sigma_e: float):
    """
    Structure tensor with coherence map.
    img : H×W float64 [0,1]
    Returns Jxx, Jxy, Jyy, coherence (all H×W)
    """
    img = img.astype(np.float64)
    Gy, Gx = np.gradient(img)  # note: np.gradient returns row-grad first

    def smooth(x, s):
        return gaussian_filter(x, sigma=s)

    Jxx = smooth(smooth(Gx * Gx, sigma_i), sigma_e)
    Jxy = smooth(smooth(Gx * Gy, sigma_i), sigma_e)
    Jyy = smooth(smooth(Gy * Gy, sigma_i), sigma_e)

    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    l1 = 0.5 * (Jxx + Jyy + tmp)
    l2 = 0.5 * (Jxx + Jyy - tmp)
    coh = (l1 - l2) ** 2 / ((l1 + l2) ** 2 + 1e-12)
    coh = np.clip(coh, 0.0, 1.0)
    return Jxx, Jxy, Jyy, coh


def vibrance_boost(img: np.ndarray, amount: float) -> np.ndarray:
    """Boost under-saturated colours (leave greys alone)."""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] / 255.0  # normalise S to [0,1]
    S = hsv[..., 1]
    hsv[..., 1] = np.clip(S + amount * (1 - S), 0.0, 1.0)
    hsv[..., 1] = (hsv[..., 1] * 255).astype(np.float32)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL).astype(np.float64) / 255.0
    return np.clip(out, 0.0, 1.0)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """img: H×W×3 float [0,1] → H×W×3 HSV [0,1]"""
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV) / np.array([360.0, 1.0, 1.0])


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """img: H×W×3 float [0,1] → H×W×3 LAB"""
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB).astype(np.float64)
    # OpenCV LAB: L in [0,255] → [0,100], a/b in [0,255] → [-128,127]
    lab[..., 0] = lab[..., 0] * 100.0 / 255.0
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """H×W×3 LAB → H×W×3 float [0,1]"""
    lab_cv = lab.copy()
    lab_cv[..., 0] = np.clip(lab_cv[..., 0] * 255.0 / 100.0, 0, 255)
    lab_cv[..., 1] = np.clip(lab_cv[..., 1] + 128.0, 0, 255)
    lab_cv[..., 2] = np.clip(lab_cv[..., 2] + 128.0, 0, 255)
    lab_cv = lab_cv.astype(np.uint8)
    rgb = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """img: H×W×3 float [0,1] → H×W float [0,1]"""
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def img_metrics(img: np.ndarray):
    """Entropy and average gradient of an image."""
    gray = (rgb_to_gray(img) * 255).astype(np.uint8)
    # Entropy via histogram
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    ev = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    # Average gradient
    gy, gx = np.gradient(gray.astype(np.float64))
    ag = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    return ev, ag


def adapthisteq(L: np.ndarray, clip_limit: float = 0.01,
                num_tiles: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE on a [0,1] luminance channel.
    Returns [0,1] array.
    """
    L8 = (np.clip(L, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit * 255,
                             tileGridSize=num_tiles)
    L8c = clahe.apply(L8)
    return L8c.astype(np.float64) / 255.0


def entropy_filter(gray_uint8: np.ndarray, radius: int = 4) -> np.ndarray:
    """Local entropy filter on uint8 grayscale image."""
    return rank_entropy(gray_uint8, disk(radius)).astype(np.float64)


def detect_sky(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Detect sky / bright uniform regions.
    Returns boolean mask H×W.
    """
    hsv = rgb_to_hsv(img)  # [0,1]
    S = hsv[..., 1]
    V = hsv[..., 2]

    # Spatial prior: top 40% strongly favoured
    sp = np.zeros((h, w), dtype=np.float64)
    sp[:round(h * 0.40), :] = 1.0
    sp[round(h * 0.40):round(h * 0.60), :] = 0.30

    # Smoothness: low gradient
    gray = rgb_to_gray(img)
    gy, gx = np.gradient(gray)
    grad_mag = gaussian_filter(np.sqrt(gx ** 2 + gy ** 2), sigma=4)
    sm = grad_mag < 0.07

    prob = (V > 0.50) & (S < 0.42)
    prob = prob.astype(np.float64) * (0.5 + 0.5 * sp) * sm.astype(np.float64)

    sky = prob > 0.27
    sky = opening(sky, disk(3))
    sky = closing(sky, disk(10))

    # Fill holes (label-based)
    from scipy.ndimage import binary_fill_holes
    sky = binary_fill_holes(sky)

    # Remove small blobs
    sky = remove_small_objects(sky, min_size=round(h * w * 0.003))
    return sky.astype(bool)


def imerode_patch(img: np.ndarray, patch: int) -> np.ndarray:
    """Min-filter (morphological erosion) with a square patch."""
    kernel = np.ones((patch, patch), dtype=np.uint8)
    # OpenCV erode works on uint8; use float trick
    img32 = img.astype(np.float32)
    out = cv2.erode(img32, kernel)
    return out.astype(np.float64)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def find_image() -> str:
    """Find an image file in the current directory."""
    supported = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                 '.gif', '.pgm', '.ppm', '.pbm', '.webp']
    priority_stems = ['foggy_image', 'input', 'hazy_image', 'foggy', 'haze',
                      'mist', 'dust', 'sand', 'test', 'image', 'img', 'photo', 'scene']

    for stem in priority_stems:
        for ext in supported:
            p = stem + ext
            if os.path.isfile(p):
                return p

    for ext in supported:
        files = glob.glob('*' + ext)
        if files:
            return files[0]

    return ''


def load_image(img_path: str) -> np.ndarray:
    """Load image using PIL, return H×W×3 float64 [0,1] RGB."""
    pil = Image.open(img_path)
    # Handle animated / multi-frame
    try:
        pil.seek(0)
    except EOFError:
        pass

    # Convert palette images
    if pil.mode == 'P':
        pil = pil.convert('RGBA')
    # Drop alpha
    if pil.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        pil = bg
    if pil.mode != 'RGB':
        pil = pil.convert('RGB')

    img = np.array(pil).astype(np.float64) / 255.0
    return img


def dehaze(img_path: str = None):
    print('=' * 65)
    print('  ULTRA DEHAZE v5 — SAND/DUST/FOG/MIST/HAZE SPECIALIST')
    print('=' * 65)

    # ------------------------------------------------------------------
    # [1] UNIVERSAL IMAGE LOADER
    # ------------------------------------------------------------------
    if img_path is None:
        img_path = find_image()
    if not img_path:
        raise FileNotFoundError(
            f'No image found in {os.getcwd()}. '
            'Supported: .jpg .png .bmp .tif .gif .pgm .ppm .webp')

    print(f'Input : {img_path}')
    I_raw = load_image(img_path)
    h0, w0 = I_raw.shape[:2]

    # Resize to max 1000px
    max_dim = max(I_raw.shape[:2])
    if max_dim > 1000:
        sc = 1000 / max_dim
        new_h = int(round(h0 * sc))
        new_w = int(round(w0 * sc))
        I_raw = cv2.resize(I_raw, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f'  Resized {h0}×{w0} → {new_h}×{new_w}')

    I = np.clip(I_raw, 0.0, 1.0)
    h, w = I.shape[:2]
    print(f'  Size : {h}×{w} px\n')

    # ------------------------------------------------------------------
    # [2] DEGRADATION CLASSIFIER
    # ------------------------------------------------------------------
    print('[2/13] Degradation classification...')

    R, G, B = I[..., 0], I[..., 1], I[..., 2]
    avg_R, avg_G, avg_B = R.mean(), G.mean(), B.mean()

    hsv_I = rgb_to_hsv(I)
    avg_S = hsv_I[..., 1].mean()
    avg_V = hsv_I[..., 2].mean()
    gray = rgb_to_gray(I)
    avg_L = gray.mean()
    std_L = gray.std()
    fog_idx = min(avg_L / (std_L + 0.05), 10.0)

    rg_ratio = avg_R / (avg_G + 1e-9)
    rb_ratio = avg_R / (avg_B + 1e-9)

    is_sandust = (rg_ratio > 1.04) and (rb_ratio > 1.15) and (avg_S < 0.30)
    is_mist = avg_B > avg_R + 0.02
    is_dense = fog_idx > 3.0

    print(f'  R={avg_R:.3f} G={avg_G:.3f} B={avg_B:.3f} | '
          f'rg={rg_ratio:.2f} rb={rb_ratio:.2f}')
    print(f'  fog_idx={fog_idx:.2f} | sand-dust={int(is_sandust)} | '
          f'mist={int(is_mist)} | dense={int(is_dense)}')

    # ------------------------------------------------------------------
    # [3] BILATERAL PRE-DENOISE
    # ------------------------------------------------------------------
    print('[3/13] Bilateral pre-denoise...')
    sigma_s = ternary(is_sandust, 2.0, 1.5)
    I_f = bilateral_approx(I, sigma_s, 0.08)

    # ------------------------------------------------------------------
    # [4] CHANNEL-SPECIFIC ATMOSPHERIC LIGHT
    # ------------------------------------------------------------------
    print('[4/13] Channel-specific atmospheric light...')

    n_top = max(round(h * w * 0.001), 20)
    A = np.zeros(3)
    for c in range(3):
        ch_flat = I_f[..., c].ravel()
        idx = np.argsort(ch_flat)[::-1]
        A[c] = ch_flat[idx[:n_top]].mean() * 0.96

    if is_sandust:
        A[0] = max(A[0], A[1])          # R >= G
        A[1] = max(A[1], A[2])          # G >= B
        A[2] = min(A[2], A[1] * 0.90)   # B noticeably lower

    A = np.clip(A, 0.50, 0.98)
    print(f'  A = [R={A[0]:.4f}  G={A[1]:.4f}  B={A[2]:.4f}]')

    # ------------------------------------------------------------------
    # [5] REGION-ADAPTIVE TEXTURE PROBABILITY MAP
    # ------------------------------------------------------------------
    print('[5/13] Region-adaptive texture map...')

    gray_uint8 = (rgb_to_gray(I_f) * 255).astype(np.uint8)
    ent_map = entropy_filter(gray_uint8, radius=4).astype(np.float64)
    ent_min, ent_max = ent_map.min(), ent_map.max()
    P_tex = (ent_map - ent_min) / (ent_max - ent_min + 1e-9)
    P_tex = gaussian_filter(P_tex, sigma=5)

    sky_mask = detect_sky(I_f, h, w)
    P_tex[sky_mask] *= 0.35

    # ------------------------------------------------------------------
    # [6] CHANNEL-WISE TRANSMISSION ESTIMATION
    # ------------------------------------------------------------------
    print('[6/13] Channel-wise transmission estimation...')

    patch = ternary(is_dense, 9, 7)
    omega_min = ternary(is_sandust, 0.60, 0.55)
    omega_max = ternary(is_sandust, ternary(is_dense, 0.95, 0.88), 0.90)
    omega_map = omega_min + (omega_max - omega_min) * P_tex

    # Per-channel DCP transmission
    T = np.zeros((h, w, 3))
    for c in range(3):
        norm_c = np.maximum(I_f[..., c] / (A[c] + 1e-9), 0.0)
        dark_c = imerode_patch(norm_c, patch)
        T[..., c] = 1.0 - omega_map * dark_c

    # Saturation-based transmission
    t_sat = 1.0 - (1.0 - avg_S) * np.maximum(omega_map - 0.1, 0.0)
    t_sat = gaussian_filter(t_sat, sigma=3)

    # Fuse
    t_min_val = ternary(is_sandust, 0.10, 0.05)

    if is_sandust:
        t_fused = T.copy()
    else:
        t_min_ch = T.min(axis=2)
        t_mono = 0.75 * t_min_ch + 0.25 * t_sat
        t_fused = np.stack([t_mono, t_mono, t_mono], axis=-1)

    for c in range(3):
        t_fused[..., c] = np.clip(t_fused[..., c], t_min_val, 0.97)

    # ------------------------------------------------------------------
    # [7] GUIDED FILTER TRANSMISSION REFINEMENT
    # ------------------------------------------------------------------
    print('[7/13] Guided-filter transmission refinement...')

    gf_r = max(round(min(h, w) * 0.03), 8)
    gf_eps = ternary(is_sandust, 1.5e-3, 7e-4)
    guide = rgb_to_gray(I_f)

    T_ref = np.zeros((h, w, 3))
    for c in range(3):
        T_ref[..., c] = guided_filter_fast(guide, t_fused[..., c], gf_r, gf_eps)
        T_ref[..., c] = np.clip(T_ref[..., c], t_min_val, 0.97)

    # ------------------------------------------------------------------
    # [8] RED-OFFSET CORRECTED SCENE RECOVERY
    # ------------------------------------------------------------------
    print('[8/13] Red-offset corrected scene recovery...')

    d_offset = np.zeros(3)
    if is_sandust:
        for c in range(3):
            d_offset[c] = avg_R - I_f[..., c].mean()
        d_offset = np.clip(d_offset, -0.05, 0.20)

    A_eff = np.clip(A - d_offset, 0.30, 0.98)
    print(f'  Offsets d=[{d_offset[0]:.4f} {d_offset[1]:.4f} {d_offset[2]:.4f}]  '
          f'A_eff=[{A_eff[0]:.4f} {A_eff[1]:.4f} {A_eff[2]:.4f}]')

    J = np.zeros_like(I)
    for c in range(3):
        J[..., c] = (I[..., c] - A_eff[c]) / T_ref[..., c] + A_eff[c]
    J = np.clip(J, 0.0, 1.0)

    # ------------------------------------------------------------------
    # [9] BLUE CHANNEL COMPENSATION
    # ------------------------------------------------------------------
    print('[9/13] Blue channel compensation...')

    if is_sandust:
        J_R, J_G, J_B = J[..., 0], J[..., 1], J[..., 2]
        ref_mean = (J_R.mean() + J_G.mean()) / 2.0
        blue_deficit = max(ref_mean - J_B.mean(), 0.0)
        alpha = min(blue_deficit * 2.5, 0.45)
        print(f'  blue_deficit={blue_deficit:.4f}  alpha={alpha:.4f}')

        J_B_comp = np.clip(J_B + alpha * (J_R + J_G) / 2.0, 0.0, 1.0)
        J[..., 2] = J_B_comp

        # LAB colour balance
        lab = rgb_to_lab(J)
        a_mean = lab[..., 1].mean()
        lab[..., 1] -= 0.4 * a_mean
        b_mean = lab[..., 2].mean()
        lab[..., 2] -= 0.35 * b_mean
        J = lab_to_rgb(lab)

    if is_mist and not is_sandust:
        lab = rgb_to_lab(J)
        lab[..., 1] = gaussian_filter(lab[..., 1], sigma=1.0)
        lab[..., 2] = gaussian_filter(lab[..., 2], sigma=1.0)
        J = lab_to_rgb(lab)

    # ------------------------------------------------------------------
    # [10] LAB ADAPTIVE GAMMA CONTRAST STRETCH
    # ------------------------------------------------------------------
    print('[10/13] LAB adaptive gamma contrast stretch...')

    lab = rgb_to_lab(J)
    L = lab[..., 0] / 100.0
    L = np.maximum(L, 1e-4)

    L_local = np.maximum(gaussian_filter(L, sigma=15), 0.01)
    gamma_px = np.log(0.5) / np.log(L_local + 1e-9)
    gamma_px = np.clip(gamma_px, 0.45, 1.80)

    L_gamma = np.clip(L ** gamma_px, 0.0, 1.0)

    target_mean_L = 0.45
    cur_mean = L_gamma.mean()
    if cur_mean > 0.01:
        scale_L = np.clip(target_mean_L / cur_mean, 0.70, 1.40)
        L_gamma = np.clip(L_gamma * scale_L, 0.0, 1.0)

    lab[..., 0] = L_gamma * 100.0

    sat_boost = ternary(is_sandust, 1.25, 1.18)
    lab[..., 1] = np.clip(lab[..., 1] * sat_boost, -128, 127)
    lab[..., 2] = np.clip(lab[..., 2] * sat_boost, -128, 127)

    J = lab_to_rgb(lab)

    # ------------------------------------------------------------------
    # [11] CLAHE (luminance only)
    # ------------------------------------------------------------------
    print('[11/13] Luminance CLAHE...')

    clip_lim = ternary(is_sandust, 0.008, 0.012)
    lab = rgb_to_lab(J)
    L_c = adapthisteq(lab[..., 0] / 100.0, clip_limit=clip_lim,
                      num_tiles=(8, 8))
    lab[..., 0] = L_c * 100.0
    J = lab_to_rgb(lab)

    # ------------------------------------------------------------------
    # [12] STRUCTURE-TENSOR COHERENCE-GATED SHARPENING
    # ------------------------------------------------------------------
    print('[12/13] Structure-tensor edge sharpening...')

    gray_J = rgb_to_gray(J)
    _, _, _, coh = structure_tensor(gray_J, sigma_i=1.5, sigma_e=3.0)

    max_sharp = ternary(is_sandust, 0.20, 0.32)
    sharp_mask = gaussian_filter(coh * max_sharp, sigma=1.0)

    J_blur = np.stack([gaussian_filter(J[..., c], sigma=0.85) for c in range(3)], axis=-1)
    for c in range(3):
        J[..., c] = J[..., c] + sharp_mask * (J[..., c] - J_blur[..., c])
    J = np.clip(J, 0.0, 1.0)

    # ------------------------------------------------------------------
    # [13] FINAL VIBRANCE + OUTPUT
    # ------------------------------------------------------------------
    print('[13/13] Final polish & save...')

    vib = ternary(is_sandust, 0.10, 0.14)
    J = vibrance_boost(J, vib)

    J = np.stack([gaussian_filter(J[..., c], sigma=0.35) for c in range(3)], axis=-1)
    J = np.clip(J, 0.0, 1.0)

    # ---- SAVE -----------------------------------------------------------
    out_dir = 'defogging_output'
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    result_path = os.path.join(out_dir, f'{stem}_dehazed.png')
    cmp_path = os.path.join(out_dir, f'{stem}_comparison.png')

    J_save = (np.clip(J, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(J_save, 'RGB').save(result_path)
    print(f'\n  Saved: {result_path}')

    # ---- METRICS --------------------------------------------------------
    ei, gi = img_metrics(I)
    ej, gj = img_metrics(J)

    # SSIM (channel-wise mean)
    sv_vals = []
    for c in range(3):
        sv_c, _ = ssim_func(I[..., c], J[..., c], full=True,
                            data_range=1.0)
        sv_vals.append(sv_c)
    sv = float(np.mean(sv_vals))

    ms = np.mean((J - I) ** 2)
    psnr_v = 10.0 * math.log10(1.0 / (ms + 1e-12))

    # ---- COMPARISON FIGURE ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), facecolor='white')

    axes[0].imshow(np.clip(I, 0, 1))
    axes[0].set_title(
        f'ORIGINAL IMAGE\nEntropy:{ei:.3f}  |  Avg Grad:{gi:.2f}',
        fontsize=12, fontweight='bold', color=(0.2, 0.2, 0.2))
    axes[0].axis('off')

    axes[1].imshow(np.clip(J, 0, 1))
    pct_e = 100 * (ej - ei) / (ei + 1e-9)
    pct_g = 100 * (gj - gi) / (gi + 1e-9)
    axes[1].set_title(
        f'DEHAZED (v5 Sand-Dust Specialist)\n'
        f'Entropy:{ej:.3f}({pct_e:+.1f}%)  '
        f'Grad:{gj:.2f}({pct_g:+.1f}%)  '
        f'SSIM:{sv:.3f}  PSNR:{psnr_v:.1f}dB',
        fontsize=11, fontweight='bold', color=(0, 0.42 * 255 / 255, 0.08 * 255 / 255))
    axes[1].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(cmp_path, dpi=200, bbox_inches='tight')
        print(f'  Comparison: {cmp_path}')
    except Exception as e:
        print(f'  Warning: comparison figure save failed — {e}')
    plt.close(fig)

    # ---- QUALITY REPORT -------------------------------------------------
    std_orig = rgb_to_gray(I).std()
    std_out = rgb_to_gray(J).std()

    print()
    print('=' * 65)
    print('                   QUALITY REPORT  (v5)')
    print('=' * 65)
    print(f'{"Metric":<26} | {"Original":>8} | {"Restored":>8} | Change')
    print('-' * 65)
    print(f'{"Entropy":<26} | {ei:8.4f} | {ej:8.4f} | {pct_e:+.2f}%')
    print(f'{"Avg Gradient":<26} | {gi:8.2f} | {gj:8.2f} | {pct_g:+.1f}%')
    print(f'{"Contrast (StdDev)":<26} | {std_orig:8.4f} | {std_out:8.4f}')
    print(f'{"SSIM":<26} |   1.0000 | {sv:8.4f}')
    print(f'{"PSNR":<26} |      --- | {psnr_v:6.2f} dB')
    print('-' * 65)
    mode = 'SAND-DUST' if is_sandust else 'GENERIC FOG'
    print(f'Mode: {mode} | fog_idx={fog_idx:.2f} | '
          f'omega=[{omega_min:.2f},{omega_max:.2f}]')
    print('=' * 65)
    print(f'✓  {result_path}')
    print(f'✓  {cmp_path}')
    print('=' * 65)

    return J, result_path, cmp_path


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    dehaze(img_path)
