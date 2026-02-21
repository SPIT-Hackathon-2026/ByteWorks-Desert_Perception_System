"""
=========================================================================
 IMAGE DEGRADATION GENERATOR
 Creates 3 degraded copies of an input image simulating:
   1. DENSE FOG     — uniform grey-white scattering veil
   2. SAND-DUST     — yellow-brown asymmetric channel degradation
   3. MIST / HAZE   — blue-tinted low-contrast soft veil

 Each variant is physically motivated to match the degradation models
 described in Ultra Dehaze v5, so the output pairs are ideal test cases
 for that pipeline.

 DEPENDENCIES:
   pip install numpy opencv-python-headless pillow scipy matplotlib

 USAGE:
   python create_degraded_images.py [image_path]
   (or drop an image in the working directory and run without arguments)

 OUTPUT:
   degraded_output/
     <name>_fog.png
     <name>_dust.png
     <name>_mist.png
     <name>_comparison.png
=========================================================================
"""

import sys
import os
import glob

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def find_image() -> str:
    supported = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                 '.gif', '.pgm', '.ppm', '.webp']
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


def load_image(path: str) -> np.ndarray:
    """Load any image → H×W×3 float64 [0,1] RGB."""
    pil = Image.open(path)
    try:
        pil.seek(0)
    except EOFError:
        pass
    if pil.mode == 'P':
        pil = pil.convert('RGBA')
    if pil.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        pil = bg
    if pil.mode != 'RGB':
        pil = pil.convert('RGB')
    return np.array(pil).astype(np.float64) / 255.0


def save_image(img: np.ndarray, path: str):
    """Save float [0,1] RGB array as PNG."""
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, 'RGB').save(path)


def resize_max(img: np.ndarray, max_dim: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    sc = max_dim / max(h, w)
    new_h, new_w = int(round(h * sc)), int(round(w * sc))
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    return np.array(pil).astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# PHYSICAL DEGRADATION MODELS
# ---------------------------------------------------------------------------

def apply_fog(img: np.ndarray, intensity: float = 0.70) -> np.ndarray:
    """
    Dense fog / thick mist simulation.
    Physical model:  I(x) = J(x)·t(x) + A·(1 - t(x))
    where:
      - A = atmospheric light ≈ [0.95, 0.95, 0.95]  (near-white in dense fog)
      - t(x) = exp(-β·d(x))   approximated from a depth map estimated via
               luminance inversion (bright pixels assumed nearer)
      - β   = extinction coefficient scaled by `intensity`

    Result: grey-white uniform veil, equal across all channels.
    """
    h, w = img.shape[:2]

    # ---- Depth proxy from inverse luminance --------------------------------
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # Smooth depth map — far objects are darker (in typical outdoor scenes)
    # Invert: bright ≈ near, dark ≈ far
    depth = 1.0 - gray
    depth = gaussian_filter(depth, sigma=max(h, w) * 0.02)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    # Add vertical gradient bias (sky / top = far)
    vert = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
    depth = 0.60 * depth + 0.40 * vert
    depth = np.clip(depth, 0, 1)

    # ---- Transmission map --------------------------------------------------
    beta = intensity * 2.5          # extinction coefficient
    t = np.exp(-beta * depth)       # Beer-Lambert law
    t = np.clip(t, 0.05, 0.95)     # avoid complete washout
    t = t[..., None]                # broadcast over channels

    # ---- Atmospheric light (fog is near-white) ------------------------------
    A = np.array([0.93, 0.93, 0.95])   # very slight blue tint — real fog

    # ---- Haze equation -------------------------------------------------------
    I_fog = img * t + A * (1.0 - t)

    # ---- Add mild Gaussian noise (fog scatters → sensor noise) ---------------
    noise_std = 0.012 * intensity
    noise = np.random.normal(0, noise_std, img.shape)
    I_fog = np.clip(I_fog + noise, 0, 1)

    return I_fog


def apply_dust(img: np.ndarray, intensity: float = 0.72) -> np.ndarray:
    """
    Sand-dust / sandstorm simulation.
    Physical model (Cheng et al. IEEE Access 2020; Gao IEEE Photonics 2020):
      - Mie scattering asymmetry: blue channel is absorbed 3-5× more than red.
      - Atmospheric light is strongly yellow-orange: A_R > A_G >> A_B.
      - Channel-specific transmission: t_R > t_G > t_B.
      - Red offset d^c = mean(I_R) - mean(I_c) → reddish global shift.
      - Low global saturation, heavy yellow-brown cast.
    """
    h, w = img.shape[:2]

    # ---- Depth proxy (same as fog, with spatial noise for turbulence) --------
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    depth = 1.0 - gray
    # Add turbulence noise to mimic uneven dust density
    turb = np.random.normal(0, 0.12, (h, w))
    turb = gaussian_filter(turb, sigma=max(h, w) * 0.04)
    turb = (turb - turb.min()) / (turb.max() - turb.min() + 1e-9) - 0.5
    depth = depth + 0.25 * turb
    depth = gaussian_filter(depth, sigma=max(h, w) * 0.015)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    vert = np.linspace(0.8, 0.2, h)[:, None] * np.ones((1, w))   # less sky bias for dust
    depth = 0.55 * depth + 0.45 * vert
    depth = np.clip(depth, 0, 1)

    # ---- Channel-specific extinction coefficients (blue >> red) ---------------
    beta_R = intensity * 1.4
    beta_G = intensity * 2.0
    beta_B = intensity * 3.5   # blue absorbed most — dust/Mie asymmetry

    t_R = np.clip(np.exp(-beta_R * depth), 0.10, 0.95)[..., None]
    t_G = np.clip(np.exp(-beta_G * depth), 0.08, 0.93)[..., None]
    t_B = np.clip(np.exp(-beta_B * depth), 0.05, 0.88)[..., None]
    T = np.concatenate([t_R, t_G, t_B], axis=-1)   # H×W×3

    # ---- Atmospheric light: yellow-orange sand colour -------------------------
    A = np.array([0.92, 0.80, 0.52])   # strongly yellow (R high, B low)

    # ---- Per-channel haze equation -------------------------------------------
    I_dust = img * T + A * (1.0 - T)

    # ---- Global red-offset shift (Kim 2021 / Springer 2016) ------------------
    # Add a uniform red-channel offset to simulate the global reddish cast
    red_offset = 0.08 * intensity
    I_dust[..., 0] = np.clip(I_dust[..., 0] + red_offset, 0, 1)

    # ---- Mild desaturation (sand scenes appear washed-out) -------------------
    gray_out = (0.299 * I_dust[..., 0] + 0.587 * I_dust[..., 1]
                + 0.114 * I_dust[..., 2])[..., None]
    desat = 0.30 * intensity
    I_dust = (1.0 - desat) * I_dust + desat * gray_out

    # ---- Coarse-grained noise (sand particles → shot noise) ------------------
    noise_std = 0.018 * intensity
    noise = np.random.normal(0, noise_std, img.shape)
    I_dust = np.clip(I_dust + noise, 0, 1)

    return I_dust


def apply_mist(img: np.ndarray, intensity: float = 0.62) -> np.ndarray:
    """
    Light mist / haze simulation.
    Characteristics (vs dense fog):
      - Atmospheric light is slightly bluish (Rayleigh scattering).
      - Transmission is higher — objects not fully washed out.
      - Soft, spatially smooth veil — mist has larger droplets.
      - Contrast and saturation reduced.
      - Slight blue channel excess: avg_B > avg_R (classifier trigger in v5).
    """
    h, w = img.shape[:2]

    # ---- Smooth depth map (mist is spatially uniform → smoother than dust) ---
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    depth = 1.0 - gray
    depth = gaussian_filter(depth, sigma=max(h, w) * 0.04)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    vert = np.linspace(1.0, 0.1, h)[:, None] * np.ones((1, w))
    depth = 0.50 * depth + 0.50 * vert
    depth = np.clip(depth, 0, 1)

    # ---- Transmission: higher than fog (mist is thinner) ---------------------
    beta = intensity * 1.6
    t = np.clip(np.exp(-beta * depth), 0.20, 0.97)
    t = t[..., None]

    # ---- Atmospheric light: cool blue-white (Rayleigh) -----------------------
    A = np.array([0.88, 0.91, 0.97])   # B slightly dominant — avg_B > avg_R

    # ---- Haze equation -------------------------------------------------------
    I_mist = img * t + A * (1.0 - t)

    # ---- Colour temperature shift toward cool blue ----------------------------
    I_mist[..., 0] = np.clip(I_mist[..., 0] * (1.0 - 0.04 * intensity), 0, 1)
    I_mist[..., 2] = np.clip(I_mist[..., 2] * (1.0 + 0.03 * intensity), 0, 1)

    # ---- Low-frequency contrast compression (mist softens edges) ------------
    blurred = gaussian_filter(I_mist, sigma=[max(h, w) * 0.005, max(h, w) * 0.005, 0])
    blend = 0.15 * intensity
    I_mist = (1.0 - blend) * I_mist + blend * blurred

    # ---- Very fine noise (water droplets → speckle) --------------------------
    noise_std = 0.008 * intensity
    noise = np.random.normal(0, noise_std, img.shape)
    I_mist = np.clip(I_mist + noise, 0, 1)

    return I_mist


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(img_path: str = None):
    print('=' * 65)
    print('  IMAGE DEGRADATION GENERATOR')
    print('  Producing: FOG / SAND-DUST / MIST variants')
    print('=' * 65)

    # ---- Find / load image --------------------------------------------------
    if img_path is None:
        img_path = find_image()
    if not img_path:
        raise FileNotFoundError(
            f'No image found in {os.getcwd()}.\n'
            'Supported: .jpg .png .bmp .tif .gif .pgm .ppm .webp')

    print(f'Input : {img_path}')
    img = load_image(img_path)
    img = resize_max(img, max_dim=1000)
    h, w = img.shape[:2]
    print(f'Size  : {h}×{w} px\n')

    # ---- Apply degradations -------------------------------------------------
    np.random.seed(42)   # reproducible noise

    print('Generating fog variant    ...', end=' ', flush=True)
    img_fog  = apply_fog(img,  intensity=0.70)
    print('done')

    print('Generating sand-dust variant ...', end=' ', flush=True)
    img_dust = apply_dust(img, intensity=0.72)
    print('done')

    print('Generating mist variant   ...', end=' ', flush=True)
    img_mist = apply_mist(img, intensity=0.62)
    print('done\n')

    # ---- Save outputs -------------------------------------------------------
    out_dir = 'degraded_output'
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    paths = {
        'fog':  os.path.join(out_dir, f'{stem}_fog.png'),
        'dust': os.path.join(out_dir, f'{stem}_dust.png'),
        'mist': os.path.join(out_dir, f'{stem}_mist.png'),
    }
    save_image(img_fog,  paths['fog'])
    save_image(img_dust, paths['dust'])
    save_image(img_mist, paths['mist'])

    for label, path in paths.items():
        print(f'  Saved [{label:>4}]: {path}')

    # ---- Comparison figure --------------------------------------------------
    variants = [
        ('ORIGINAL',              img,      (0.15, 0.15, 0.15)),
        ('FOG\n(uniform grey veil, ~equal channels)', img_fog,  (0.10, 0.30, 0.60)),
        ('SAND-DUST\n(yellow cast, t_R>t_G>t_B)',     img_dust, (0.65, 0.40, 0.05)),
        ('MIST / HAZE\n(cool blue tint, soft veil)',  img_mist, (0.20, 0.50, 0.55)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white')
    fig.suptitle('Degradation Generator — Test Images for Ultra Dehaze v5',
                 fontsize=13, fontweight='bold', color=(0.1, 0.1, 0.1), y=1.01)

    for ax, (title, var_img, col) in zip(axes, variants):
        ax.imshow(np.clip(var_img, 0, 1))
        r_m = var_img[..., 0].mean()
        g_m = var_img[..., 1].mean()
        b_m = var_img[..., 2].mean()
        ax.set_title(f'{title}\nR={r_m:.3f}  G={g_m:.3f}  B={b_m:.3f}',
                     fontsize=10, fontweight='bold', color=col)
        ax.axis('off')

    plt.tight_layout()
    cmp_path = os.path.join(out_dir, f'{stem}_comparison.png')
    plt.savefig(cmp_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Comparison : {cmp_path}')

    # ---- Channel stats summary -----------------------------------------------
    print()
    print('=' * 65)
    print('  CHANNEL STATISTICS SUMMARY')
    print('=' * 65)
    header = f'{"Variant":<14} | {"mean R":>7} | {"mean G":>7} | {"mean B":>7} | {"R/G":>5} | {"R/B":>5} | {"mean S":>7}'
    print(header)
    print('-' * 65)
    for title, var_img, _ in variants:
        name = title.split('\n')[0]
        r_m = var_img[..., 0].mean()
        g_m = var_img[..., 1].mean()
        b_m = var_img[..., 2].mean()
        rg  = r_m / (g_m + 1e-9)
        rb  = r_m / (b_m + 1e-9)
        # Saturation estimate
        mx = var_img.max(axis=2); mn = var_img.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0).mean()
        print(f'{name:<14} | {r_m:7.4f} | {g_m:7.4f} | {b_m:7.4f} | {rg:5.2f} | {rb:5.2f} | {sat:7.4f}')
    print('=' * 65)
    print()
    print('Tip: feed each output directly into ultra_dehaze_v5.py')
    print('     e.g.  python ultra_dehaze_v5.py degraded_output/<name>_dust.png')
    print('=' * 65)


if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(img_path)
