"""
Satellite Streak Synthetic Dataset Generator  v3
=================================================
Generates ~8000 paired (image / mask / clean) samples for U-Net training.

Target distribution
-------------------
  starfields   : 2100  (from raw_images/starfields/)
  nebula       : 1400  (from raw_images/nebula/)
  noise_heavy  : 1400  (synthesised from existing images)
  empty        : 1400  (synthesised — near-black, low-signal)
  galaxies     : 1000  (from raw_images/galaxies/)   ← NEW in v3
  no-streak    :  ~700 (spread across all categories, mask = zeros)

NOTE: raw_images/sparse_stars/ is completely ignored.

Usage
-----
  python generate_dataset_v2.py \\
      --input_dir  raw_images/ \\
      --output_dir dataset/ \\
      --seed 42

  # Dry-run (skip saving, just count):
  python generate_dataset_v2.py --input_dir raw_images/ --dry_run

Dependencies
------------
  pip install opencv-python numpy tqdm scipy
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE   = 256
EDGE_CROP    = 0.10          # crop 10 % from each edge before random patch
MEAN_LOW_DN  = 3             # reject patch if mean (0-255) below this
MEAN_HIGH_DN = 220           # reject patch if mean (0-255) above this
NO_STREAK_FRACTION = 0.10    # ~10 % of samples have no streak
NO_STREAK_FRACTION_GALAXIES = 0.15  # galaxies: 10–20 % no-streak (midpoint)

# Category → target count
CATEGORY_TARGETS: dict[str, int] = {
    "starfields":  2100,
    "nebula":      1400,
    "noise_heavy": 1400,
    "empty":       1400,
    "galaxies":    1000,   # ← NEW
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – IMAGE LOADING & CROPPING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str | Path) -> Optional[np.ndarray]:
    """Load image from disk as BGR uint8. Returns None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        log.warning("Could not read: %s", path)
    return img


def edge_crop(img: np.ndarray, frac: float = EDGE_CROP) -> Optional[np.ndarray]:
    """
    Remove `frac` of pixels from every edge (top / bottom / left / right).
    Returns None if the result is smaller than one PATCH_SIZE.
    """
    h, w = img.shape[:2]
    dy = int(frac * h)
    dx = int(frac * w)
    cropped = img[dy: h - dy, dx: w - dx]
    ch, cw = cropped.shape[:2]
    if ch < PATCH_SIZE or cw < PATCH_SIZE:
        return None
    return cropped


def extract_random_patch(
    img: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> Optional[np.ndarray]:
    """Extract one random square patch. Returns None if image is too small."""
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        return None
    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)
    return img[y: y + patch_size, x: x + patch_size].copy()


def extract_structured_patch(
    img: np.ndarray,
    patch_size: int = PATCH_SIZE,
    n_candidates: int = 8,
) -> Optional[np.ndarray]:
    """
    Probabilistic crop biased toward high-intensity / structured regions.

    Strategy
    --------
    Sample `n_candidates` random positions, score each by the variance of the
    candidate patch (variance captures both brightness and structure), then
    pick the highest-scoring one.  Falls back to a plain random crop if the
    image is too small.

    Used for the galaxies category (30 % of crops use this path).
    """
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        return None

    best_patch = None
    best_score = -1.0

    for _ in range(n_candidates):
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        candidate = img[y: y + patch_size, x: x + patch_size]
        # score = variance × mean  →  rewards bright AND structured regions
        score = float(np.var(candidate)) * float(np.mean(candidate))
        if score > best_score:
            best_score = score
            best_patch = candidate.copy()

    return best_patch


def to_gray_float(patch: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 patch to float32 grayscale in [0, 1]."""
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    return gray.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – NOISE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def _sample_sigma(tier: str) -> float:
    """Draw a noise sigma consistent with the requested tier."""
    if tier == "low":
        return random.uniform(2, 6)
    if tier == "medium":
        return random.uniform(6, 14)
    return random.uniform(14, 30)   # high


def _noise_tier(category: str) -> str:
    """Sample a noise tier according to per-category probability weights."""
    if category == "noise_heavy":
        # mostly high, some medium
        return random.choices(["low", "medium", "high"], weights=[5, 25, 70])[0]
    if category == "empty":
        return random.choices(["low", "medium", "high"], weights=[70, 25, 5])[0]
    if category == "nebula":
        return random.choices(["low", "medium", "high"], weights=[40, 50, 10])[0]
    if category == "galaxies":
        # moderate noise — similar to nebula but slightly lower floor
        return random.choices(["low", "medium", "high"], weights=[45, 45, 10])[0]
    # starfields
    return random.choices(["low", "medium", "high"], weights=[70, 25, 5])[0]


def add_gaussian_noise(img: np.ndarray, sigma_dn: float) -> np.ndarray:
    """
    Add zero-mean Gaussian noise.
    `sigma_dn` is in DN units (0-255); image is float [0,1].
    """
    sigma_f = sigma_dn / 255.0
    noise   = np.random.normal(0.0, sigma_f, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def add_poisson_noise(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Simulate photon shot noise (Poisson process).
    `scale` controls effective exposure; higher = less noisy.
    Input is clipped to [0, 1] before sampling — np.random.poisson
    requires lam >= 0 and will raise ValueError on negative values.
    """
    scale = max(scale, 0.1)
    lam   = np.clip(img, 0.0, 1.0) * 255.0 * scale   # guarantee lam >= 0
    noisy = np.random.poisson(lam).astype(np.float32)
    noisy /= (255.0 * scale)
    return np.clip(noisy, 0.0, 1.0)


def apply_noise(img: np.ndarray, category: str) -> np.ndarray:
    """
    Apply a random combination of Gaussian and/or Poisson noise
    appropriate for the given category.
    """
    tier    = _noise_tier(category)
    sigma   = _sample_sigma(tier)
    roll    = random.random()

    if category == "noise_heavy":
        # always Gaussian + often Poisson
        img = add_gaussian_noise(img, sigma)
        if random.random() < 0.70:
            img = add_poisson_noise(img, scale=random.uniform(0.3, 1.0))

    elif roll < 0.50:
        # Gaussian only
        img = add_gaussian_noise(img, sigma)
    elif roll < 0.75:
        # Poisson only
        img = add_poisson_noise(img, scale=random.uniform(1.0, 5.0))
    else:
        # Both
        img = add_gaussian_noise(img, sigma * 0.6)
        img = add_poisson_noise(img, scale=random.uniform(1.0, 3.0))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – GRADIENT MODELS
# ══════════════════════════════════════════════════════════════════════════════

def _linear_gradient(size: int) -> np.ndarray:
    """Directional linear gradient at a random angle, strength ∈ [0.02, 0.12]."""
    angle   = random.uniform(0, 2 * math.pi)
    amp     = random.uniform(0.02, 0.12)
    xs      = np.linspace(-1, 1, size, dtype=np.float32)
    ys      = np.linspace(-1, 1, size, dtype=np.float32)
    xx, yy  = np.meshgrid(xs, ys)
    g       = amp * (math.cos(angle) * xx + math.sin(angle) * yy)
    return g


def _radial_gradient(size: int) -> np.ndarray:
    """Radial vignette centred at a random point, strength ∈ [0.03, 0.15]."""
    cx  = random.uniform(0.2, 0.8)
    cy  = random.uniform(0.2, 0.8)
    amp = random.uniform(0.03, 0.15)
    xs  = np.linspace(0, 1, size, dtype=np.float32)
    ys  = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist   = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist  /= dist.max() + 1e-6
    # invert so centre is brighter
    g = amp * (1.0 - dist)
    return g


def _patchy_gradient(size: int) -> np.ndarray:
    """
    Smooth blob / patchy gradient: random low-frequency noise blurred heavily.
    Simulates uneven sky illumination or nebula structure.
    """
    amp    = random.uniform(0.02, 0.10)
    # start from coarse noise, then blur
    coarse = np.random.uniform(-1, 1, (size // 8, size // 8)).astype(np.float32)
    coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    sigma  = random.uniform(20, 60)
    smooth = gaussian_filter(coarse, sigma=sigma / (size / 256))
    smooth -= smooth.mean()
    mx     = np.abs(smooth).max() + 1e-6
    return (smooth / mx * amp).astype(np.float32)


def add_gradient(img: np.ndarray, category: str) -> np.ndarray:
    """
    Randomly select and apply one of {linear, radial, patchy, none} gradients
    with per-category probability weights.
    """
    if category == "empty":
        # empty images stay uniform
        return img
    if category == "noise_heavy":
        # gradient mostly invisible under heavy noise
        if random.random() > 0.20:
            return img

    # probability weights: [linear, radial, patchy, none]
    if category == "nebula":
        weights = [25, 30, 35, 10]
    elif category == "galaxies":
        # moderate gradients; radial vignette most natural for galaxy cores
        weights = [20, 35, 30, 15]
    else:  # starfields
        weights = [20, 20, 15, 45]

    choice = random.choices(["linear", "radial", "patchy", "none"], weights=weights)[0]

    if choice == "none":
        return img
    if choice == "linear":
        g = _linear_gradient(img.shape[0])
    elif choice == "radial":
        g = _radial_gradient(img.shape[0])
    else:
        g = _patchy_gradient(img.shape[0])

    return np.clip(img + g, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – AUGMENTATION COMBINATOR
# ══════════════════════════════════════════════════════════════════════════════

def augment(img: np.ndarray, category: str) -> np.ndarray:
    """
    Apply a stochastic combination of augmentations appropriate for the
    category. Possible combinations:
      noise only | gradient only | noise + gradient | blur + noise | none
    """
    roll = random.random()

    # Category-specific contrast reduction
    if category in ("starfields", "nebula"):
        img = img * random.uniform(0.55, 0.85)
    elif category == "galaxies":
        # preserve brightness variation — galaxy cores can be relatively bright
        img = img * random.uniform(0.60, 0.90)
    elif category == "empty":
        img = img * random.uniform(0.05, 0.25)   # very dark
    elif category == "noise_heavy":
        img = img * random.uniform(0.40, 0.80)

    img = img.clip(0.0, 1.0).astype(np.float32)

    # Combination draw
    if category == "noise_heavy":
        img = apply_noise(img, category)          # always noise
        if random.random() < 0.20:
            img = add_gradient(img, category)
    elif category == "empty":
        img = apply_noise(img, category)          # slight noise only
    elif roll < 0.25:
        img = apply_noise(img, category)
    elif roll < 0.45:
        img = add_gradient(img, category)
    elif roll < 0.75:
        img = apply_noise(img, category)
        img = add_gradient(img, category)
    elif roll < 0.90:
        # blur + noise
        ksize = random.choice([3, 5])
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)
        img   = apply_noise(img, category)
    # else: no augmentation

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – SYNTHETIC IMAGE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_empty_image(size: int = PATCH_SIZE) -> np.ndarray:
    """
    Create a near-black image with very faint Gaussian noise.
    Occasional extremely faint point sources to avoid pure black.
    """
    img = np.zeros((size, size), dtype=np.float32)

    # Very faint sky glow
    base = random.uniform(0.005, 0.04)
    img += base

    # A handful of extremely faint point sources (optional)
    if random.random() < 0.50:
        n_pts = random.randint(1, 6)
        for _ in range(n_pts):
            px = random.randint(0, size - 1)
            py = random.randint(0, size - 1)
            val = random.uniform(0.04, 0.12)
            img[py, px] = val

    # Slight noise
    sigma = random.uniform(1, 5) / 255.0
    img  += np.random.normal(0, sigma, img.shape).astype(np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_noise_heavy_from(base: np.ndarray, category_src: str = "starfields") -> np.ndarray:
    """
    Start from an existing preprocessed patch and apply aggressive noise,
    preserving some underlying structure.
    """
    # Partial contrast suppression so structure is faint but present
    img = base * random.uniform(0.3, 0.6)

    # Strong Gaussian
    sigma = random.uniform(20, 45) / 255.0
    img  += np.random.normal(0, sigma, img.shape).astype(np.float32)

    # Poisson on top
    if random.random() < 0.60:
        img = add_poisson_noise(img, scale=random.uniform(0.2, 0.6))

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – STREAK & MASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_streak_mask(size: int = PATCH_SIZE) -> np.ndarray:
    """
    Produce a soft float32 mask [0,1] for a single satellite streak.

    Properties
    ----------
    • Random angle  0–180°
    • Thickness     1–3 px
    • Variable brightness along the streak
    • 35 % chance of dashed / broken segments
    • Optional final Gaussian blur (PSF)
    """
    mask    = np.zeros((size, size), dtype=np.float32)
    angle   = random.uniform(0, 180)
    rad     = math.radians(angle)
    cos_a   = math.cos(rad)
    sin_a   = math.sin(rad)

    # Length: 30–130 % of patch diagonal
    diag   = math.sqrt(2) * size
    length = int(random.uniform(0.30, 1.30) * diag)

    # Start point (may be outside frame for realism)
    margin = size // 3
    x0     = random.randint(-margin, size + margin)
    y0     = random.randint(-margin, size + margin)

    # Dashed streak parameters
    dashed      = random.random() < 0.35
    dash_period = random.randint(6, 18)
    dash_duty   = random.uniform(0.40, 0.75)

    # Brightness variation
    vary     = random.random() < 0.65
    freq     = random.uniform(0.008, 0.04)
    var_amp  = random.uniform(0.0, 0.45)

    thickness = random.randint(1, 3)

    for t in range(length):
        px = int(x0 + t * cos_a)
        py = int(y0 + t * sin_a)

        if not (0 <= px < size and 0 <= py < size):
            continue

        if dashed and (t % dash_period) / dash_period > dash_duty:
            continue

        intensity = 1.0
        if vary:
            intensity = 0.55 + 0.45 * math.sin(2 * math.pi * freq * t)
            intensity += random.uniform(-var_amp, var_amp)
            intensity = float(np.clip(intensity, 0.0, 1.0))

        for dw in range(thickness):
            # perpendicular offset
            npx = px - int(dw * sin_a)
            npy = py + int(dw * cos_a)
            if 0 <= npx < size and 0 <= npy < size:
                mask[npy, npx] = max(mask[npy, npx], intensity)

    # Soft PSF blur
    if True:
        ksize = random.choice([3, 5])
        mask  = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask.astype(np.float32)


def apply_streak(
    clean: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlay the streak on the clean image.

    Returns
    -------
    streak_img  : float32 [0,1]
    binary_mask : uint8   {0, 255}
    """
    brightness  = random.uniform(0.25, 1.0)
    streak_img  = np.clip(clean + mask * brightness, 0.0, 1.0).astype(np.float32)
    binary_mask = (mask > 0.04).astype(np.uint8) * 255
    return streak_img, binary_mask


def empty_mask(size: int = PATCH_SIZE) -> np.ndarray:
    """Return an all-zero binary mask (for no-streak images)."""
    return np.zeros((size, size), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PATCH VALIDITY
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_patch(img: np.ndarray) -> bool:
    """
    Reject patches that are blank, over-saturated, or have stars buried
    in noise (low SNR proxy).
    """
    mean_dn = img.mean() * 255.0
    if mean_dn < MEAN_LOW_DN or mean_dn > MEAN_HIGH_DN:
        return False

    std      = float(np.std(img))
    peak     = float(np.percentile(img, 95))
    mean_f   = float(np.mean(img))
    snr      = (peak - mean_f) / (std + 1e-6)
    return snr >= 0.4


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_float_image(arr: np.ndarray, path: str | Path) -> None:
    """Save float32 [0,1] as 16-bit PNG."""
    img16 = (arr * 65535).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), img16)


def save_mask_image(mask: np.ndarray, path: str | Path) -> None:
    cv2.imwrite(str(path), mask)


def collect_sources(folder: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – CATEGORY PROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def _process_real_category(
    src_files: list[Path],
    category: str,
    n_target: int,
    no_streak_quota: int,
    out: Path,
    sample_idx: int,
    dry_run: bool,
    pbar: tqdm,
) -> int:
    """
    Draw random patches from `src_files`, augment per category rules,
    add streak (or not), and save triplets.

    Returns the updated global sample_idx.
    """
    if not src_files:
        log.warning("[%s] No source files found – skipping.", category)
        return sample_idx

    generated     = 0
    no_streak_cnt = 0
    max_tries     = n_target * 15

    for attempt in range(max_tries):
        if generated >= n_target:
            break

        src    = random.choice(src_files)
        raw    = load_image(src)
        if raw is None:
            continue

        cropped = edge_crop(raw)
        if cropped is None:
            continue

        patch = extract_random_patch(cropped)
        if patch is None:
            continue

        img = to_gray_float(patch)

        if category == "noise_heavy":
            clean = generate_noise_heavy_from(img)
        else:
            clean = augment(img, category)

        if not is_valid_patch(clean):
            continue

        # Decide streak vs no-streak
        add_no_streak = (
            no_streak_cnt < no_streak_quota
            and random.random() < NO_STREAK_FRACTION * 2   # weighted draw
        )

        fname = f"{sample_idx:05d}.png"

        if not dry_run:
            save_float_image(clean, out / "clean_images" / fname)

        if add_no_streak:
            mask       = empty_mask()
            streak_img = clean.copy()
            no_streak_cnt += 1
        else:
            soft_mask  = generate_streak_mask()
            streak_img, mask = apply_streak(clean, soft_mask)

        if not dry_run:
            save_float_image(streak_img, out / "images" / fname)
            save_mask_image(mask,        out / "masks"  / fname)

        sample_idx += 1
        generated  += 1
        pbar.update(1)

    if generated < n_target:
        log.warning("[%s] Only generated %d / %d.", category, generated, n_target)

    return sample_idx


def _process_empty_category(
    src_files: list[Path],          # used as optional base; may be empty
    n_target: int,
    no_streak_quota: int,
    out: Path,
    sample_idx: int,
    dry_run: bool,
    pbar: tqdm,
) -> int:
    """
    Generate near-black empty images (fully synthetic, no real source needed).
    Optionally uses a real image as a very faint base (30 % of the time).
    """
    generated     = 0
    no_streak_cnt = 0

    for _ in range(n_target * 3):
        if generated >= n_target:
            break

        # 30 % chance: derive from a real (very dark) base
        if src_files and random.random() < 0.30:
            raw     = load_image(random.choice(src_files))
            cropped = edge_crop(raw) if raw is not None else None
            patch   = extract_random_patch(cropped) if cropped is not None else None
            if patch is not None:
                base  = to_gray_float(patch) * random.uniform(0.02, 0.08)
                noise = random.uniform(1, 4) / 255.0
                clean = np.clip(base + np.random.normal(0, noise, base.shape), 0, 1).astype(np.float32)
            else:
                clean = generate_empty_image()
        else:
            clean = generate_empty_image()

        fname = f"{sample_idx:05d}.png"
        add_no_streak = (
            no_streak_cnt < no_streak_quota
            and random.random() < NO_STREAK_FRACTION * 2
        )

        if not dry_run:
            save_float_image(clean, out / "clean_images" / fname)

        if add_no_streak:
            mask       = empty_mask()
            streak_img = clean.copy()
            no_streak_cnt += 1
        else:
            soft_mask  = generate_streak_mask()
            streak_img, mask = apply_streak(clean, soft_mask)

        if not dry_run:
            save_float_image(streak_img, out / "images" / fname)
            save_mask_image(mask,        out / "masks"  / fname)

        sample_idx += 1
        generated  += 1
        pbar.update(1)

    return sample_idx


def _process_galaxies_category(
    src_files: list[Path],
    n_target: int,
    no_streak_quota: int,
    out: Path,
    sample_idx: int,
    dry_run: bool,
    pbar: tqdm,
) -> int:
    """
    Process the galaxies category.

    Differences from _process_real_category
    ----------------------------------------
    • Probabilistic cropping:
        70 % → extract_random_patch   (uniform spatial sampling)
        30 % → extract_structured_patch (biased toward bright/structured regions)
    • No-streak fraction: NO_STREAK_FRACTION_GALAXIES (~15 %)
    • No heavy blur is applied (galaxy structure is preserved)
    • Augmentation routed through augment(..., category="galaxies")

    Everything else (edge crop, grayscale, normalise, streak, save) is identical
    to the existing pipeline.
    """
    if not src_files:
        log.warning("[galaxies] No source files found – skipping.")
        return sample_idx

    generated     = 0
    no_streak_cnt = 0
    max_tries     = n_target * 15

    for _ in range(max_tries):
        if generated >= n_target:
            break

        src = random.choice(src_files)
        raw = load_image(src)
        if raw is None:
            continue

        cropped = edge_crop(raw)
        if cropped is None:
            continue

        # ── Probabilistic crop: 70 % random, 30 % structured ─────────────────
        if random.random() < 0.70:
            patch = extract_random_patch(cropped)
        else:
            patch = extract_structured_patch(cropped)

        if patch is None:
            continue

        img   = to_gray_float(patch)
        clean = augment(img, "galaxies")

        if not is_valid_patch(clean):
            continue

        # ── Streak decision ───────────────────────────────────────────────────
        add_no_streak = (
            no_streak_cnt < no_streak_quota
            and random.random() < NO_STREAK_FRACTION_GALAXIES * 2
        )

        fname = f"{sample_idx:05d}.png"

        if not dry_run:
            save_float_image(clean, out / "clean_images" / fname)

        if add_no_streak:
            mask       = empty_mask()
            streak_img = clean.copy()
            no_streak_cnt += 1
        else:
            soft_mask  = generate_streak_mask()
            streak_img, mask = apply_streak(clean, soft_mask)

        if not dry_run:
            save_float_image(streak_img, out / "images" / fname)
            save_mask_image(mask,        out / "masks"  / fname)

        sample_idx += 1
        generated  += 1
        pbar.update(1)

    if generated < n_target:
        log.warning("[galaxies] Only generated %d / %d.", generated, n_target)

    return sample_idx


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(
    input_dir: str,
    output_dir: str,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    inp = Path(input_dir)
    out = Path(output_dir)

    # ── Output folders ────────────────────────────────────────────────────────
    if not dry_run:
        for sub in ("images", "masks", "clean_images"):
            (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Source image discovery  (sparse_stars is explicitly excluded) ─────────
    starfield_srcs   = collect_sources(inp / "starfields")
    nebula_srcs      = collect_sources(inp / "nebula")
    galaxy_srcs      = collect_sources(inp / "galaxies")   # ← NEW
    # noise_heavy and empty reuse all real images as optional bases
    all_real_srcs    = starfield_srcs + nebula_srcs + galaxy_srcs
    # sparse_stars is never touched ↑

    log.info(
        "Source counts — starfields: %d  nebula: %d  galaxies: %d  (sparse_stars: ignored)",
        len(starfield_srcs), len(nebula_srcs), len(galaxy_srcs),
    )

    # ── No-streak quota: 700 total, distributed proportionally ───────────────
    total_target     = sum(CATEGORY_TARGETS.values())   # 6300 base + 700 no-streak
    # We embed the 700 no-streak samples within each category bucket
    no_streak_total  = 700
    no_streak_alloc  = {
        cat: round(no_streak_total * n / total_target)
        for cat, n in CATEGORY_TARGETS.items()
    }

    total_samples = sum(CATEGORY_TARGETS.values())
    pbar          = tqdm(total=total_samples, desc="Generating", unit="sample")
    sample_idx    = 0

    # ── STARFIELDS ────────────────────────────────────────────────────────────
    log.info("== starfields (%d samples) ==", CATEGORY_TARGETS["starfields"])
    sample_idx = _process_real_category(
        src_files      = starfield_srcs,
        category       = "starfields",
        n_target       = CATEGORY_TARGETS["starfields"],
        no_streak_quota= no_streak_alloc["starfields"],
        out            = out,
        sample_idx     = sample_idx,
        dry_run        = dry_run,
        pbar           = pbar,
    )

    # ── NEBULA ────────────────────────────────────────────────────────────────
    log.info("== nebula (%d samples) ==", CATEGORY_TARGETS["nebula"])
    sample_idx = _process_real_category(
        src_files      = nebula_srcs,
        category       = "nebula",
        n_target       = CATEGORY_TARGETS["nebula"],
        no_streak_quota= no_streak_alloc["nebula"],
        out            = out,
        sample_idx     = sample_idx,
        dry_run        = dry_run,
        pbar           = pbar,
    )

    # ── NOISE-HEAVY (synthesised from real bases) ─────────────────────────────
    log.info("== noise_heavy (%d samples) ==", CATEGORY_TARGETS["noise_heavy"])
    sample_idx = _process_real_category(
        src_files      = all_real_srcs,
        category       = "noise_heavy",
        n_target       = CATEGORY_TARGETS["noise_heavy"],
        no_streak_quota= no_streak_alloc["noise_heavy"],
        out            = out,
        sample_idx     = sample_idx,
        dry_run        = dry_run,
        pbar           = pbar,
    )

    # ── EMPTY (fully synthetic) ───────────────────────────────────────────────
    log.info("== empty (%d samples) ==", CATEGORY_TARGETS["empty"])
    sample_idx = _process_empty_category(
        src_files      = all_real_srcs,   # optional dark base
        n_target       = CATEGORY_TARGETS["empty"],
        no_streak_quota= no_streak_alloc["empty"],
        out            = out,
        sample_idx     = sample_idx,
        dry_run        = dry_run,
        pbar           = pbar,
    )

    # ── GALAXIES ──────────────────────────────────────────────────────────────
    log.info("== galaxies (%d samples) ==", CATEGORY_TARGETS["galaxies"])
    sample_idx = _process_galaxies_category(
        src_files      = galaxy_srcs,
        n_target       = CATEGORY_TARGETS["galaxies"],
        no_streak_quota= no_streak_alloc["galaxies"],
        out            = out,
        sample_idx     = sample_idx,
        dry_run        = dry_run,
        pbar           = pbar,
    )

    pbar.close()

    _print_summary(out, sample_idx, dry_run)


def _print_summary(out: Path, n: int, dry_run: bool) -> None:
    width = 58
    print("\n" + "═" * width)
    print(f"  {'[DRY RUN] ' if dry_run else ''}Dataset generation complete  (v3 + galaxies)")
    print("═" * width)
    print(f"  Total samples   : {n:,}")
    if not dry_run:
        print(f"  images/         : {out / 'images'}")
        print(f"  masks/          : {out / 'masks'}")
        print(f"  clean_images/   : {out / 'clean_images'}")
    print("─" * width)
    print("  Distribution")
    for cat, tgt in CATEGORY_TARGETS.items():
        print(f"    {cat:<16}: {tgt:>5}")
    print(f"    {'no-streak':<16}: {'~700':>5}  (embedded)")
    print("═" * width + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – OPTIONAL VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_sample(
    clean: np.ndarray,
    streak: np.ndarray,
    mask: np.ndarray,
    title: str = "Sample",
) -> None:
    """
    Display a side-by-side preview: clean | streak | mask.
    Requires a display / GUI backend.
    """
    h = clean.shape[0]
    gap = np.ones((h, 4), dtype=np.float32)

    row = np.hstack([clean, gap, streak, gap, mask.astype(np.float32) / 255.0])
    cv2.imshow(title + "  (clean | streak | mask)  — press any key", row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_single_sample(input_dir: str, seed: int = 0) -> None:
    """
    Generate and display one sample from the starfields folder.
    Useful for quick visual QA without running the full pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)

    srcs = collect_sources(Path(input_dir) / "starfields")
    if not srcs:
        log.error("No starfield images found in %s/starfields/", input_dir)
        return

    for src in srcs:
        raw     = load_image(src)
        cropped = edge_crop(raw) if raw is not None else None
        patch   = extract_random_patch(cropped) if cropped is not None else None
        if patch is None:
            continue

        img   = to_gray_float(patch)
        clean = augment(img, "starfields")
        if not is_valid_patch(clean):
            continue

        mask_soft        = generate_streak_mask()
        streak_img, mask = apply_streak(clean, mask_soft)
        visualize_sample(clean, streak_img, mask, title="starfields")
        return

    log.warning("Could not produce a valid demo sample.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic satellite-streak dataset generator v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  "-i", required=True,
                   help="Root folder containing starfields/ and nebula/ sub-dirs.")
    p.add_argument("--output_dir", "-o", default="dataset",
                   help="Root output dir; sub-folders created automatically.")
    p.add_argument("--seed",       "-s", type=int, default=42)
    p.add_argument("--dry_run",    action="store_true",
                   help="Count samples without writing any files.")
    p.add_argument("--demo",       action="store_true",
                   help="Display one sample and exit (requires GUI).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        demo_single_sample(args.input_dir, seed=args.seed)
    else:
        build_dataset(
            input_dir  = args.input_dir,
            output_dir = args.output_dir,
            seed       = args.seed,
            dry_run    = args.dry_run,
        )