"""
Synthetic Dataset Generation Pipeline for Satellite Streak Detection
=====================================================================
Generates paired (image + mask) samples for training a U-Net segmentation model.

Pipeline:
  1. Crop watermark regions from raw starfield images
  2. Extract random 256x256 patches
  3. Convert to grayscale & simulate astronomical conditions
  4. Save clean images (no streaks)
  5. Add physically realistic satellite streaks
  6. Save streak images + binary masks

Usage:
    python generate_dataset.py --input_dir raw_images/ --output_dir dataset/ --target 2000

Dependencies:
    pip install opencv-python numpy tqdm
"""

import os
import cv2
import math
import random
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants / Defaults
# ─────────────────────────────────────────────
PATCH_SIZE        = 256
PATCHES_PER_IMAGE = (20, 50)   # (min, max) random patches per source image
MEAN_LOW          = 5          # skip patch if mean pixel value below this
MEAN_HIGH         = 200        # skip patch if mean pixel value above this


# ══════════════════════════════════════════════════════════════════════════════
# 1. CROP – remove watermark regions
# ══════════════════════════════════════════════════════════════════════════════
def crop_image(img: np.ndarray) -> np.ndarray | None:
    """
    Remove known watermark regions from raw download images:
      - Bottom 15 %  (lower watermark strip)
      - Right  45 %  (large right-side iStock watermark – aggressive crop)
      - Left    5 %  (left margin artefact)

    The right boundary is moved from 65 % → 55 % of width to fully eliminate
    the iStock overlay that was leaking into the previous crop.

    Returns the cropped image, or None if the result is too small.
    """
    h, w = img.shape[:2]

    y_end   = int(0.85 * h)
    x_start = int(0.05 * w)
    x_end   = int(0.55 * w)   # FIX 1: was 0.65 – now removes right 45 %

    cropped = img[0:y_end, x_start:x_end]

    ch, cw = cropped.shape[:2]
    if ch < PATCH_SIZE or cw < PATCH_SIZE:
        log.debug("Cropped image too small (%dx%d) – skipping.", cw, ch)
        return None

    return cropped


# ══════════════════════════════════════════════════════════════════════════════
# 2. PATCH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_patches(
    img: np.ndarray,
    n_patches: int,
    patch_size: int = PATCH_SIZE,
) -> list[np.ndarray]:
    """
    Sample `n_patches` random 256×256 crops from a (possibly large) image.
    Skips patches whose mean falls outside [MEAN_LOW, MEAN_HIGH].
    """
    h, w = img.shape[:2]
    patches = []
    max_attempts = n_patches * 10   # avoid infinite loop on degenerate images

    attempts = 0
    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        patch = img[y : y + patch_size, x : x + patch_size]

        mean_val = patch.mean()
        if mean_val < MEAN_LOW or mean_val > MEAN_HIGH:
            continue

        patches.append(patch.copy())

    if len(patches) < n_patches:
        log.debug(
            "Only collected %d/%d patches after %d attempts.",
            len(patches), n_patches, attempts,
        )

    return patches


# ══════════════════════════════════════════════════════════════════════════════
# 3 + 4 + 5. PRE-PROCESS PATCH
#   grayscale → contrast reduction → blur → gradient → noise → normalise
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """
    Convert a BGR patch to a normalised float32 grayscale array that mimics
    realistic ground-based astronomical imagery.

    Steps
    -----
    1. Convert to grayscale (uint8 0-255)
    2. Reduce contrast  (simulate sky glow / sensor compression)
    3. Slight Gaussian blur  (optical PSF)
    4. Probabilistic Gaussian noise  (70 % low / 25 % medium / 5 % high)
    5. Normalise to [0, 1]
    Note: sky gradient step is intentionally disabled – uniform dark background
    is more representative of real astronomical data at this stage.
    """
    # 3. Grayscale
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = patch.astype(np.float32)

    # 4a. Contrast reduction
    gray = gray * random.uniform(0.5, 0.8)

    # 4b. Slight optical blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4c. Sky background gradient – DISABLED
    #     Removed: directional gradient was too visually dominant and introduced
    #     background bias inconsistent with real astronomical frames.
    #     May be re-introduced later at very low amplitude if needed.
    # if random.random() < 0.75:
    #     gray = _add_sky_gradient(gray)

    # 4d. Probabilistic Gaussian noise – three tiers:
    #       70 % → LOW    σ ∈ [3,  6)   – typical well-exposed frame
    #       25 % → MEDIUM σ ∈ [6, 12)   – moderate background / short exp.
    #        5 % → HIGH   σ ∈ [12, 16]  – poor seeing / high-gain readout
    noise_roll = random.random()
    if noise_roll < 0.70:
        sigma = random.uniform(3, 6)      # LOW   (~70 % of samples)
    elif noise_roll < 0.95:
        sigma = random.uniform(6, 12)     # MEDIUM (~25 % of samples)
    else:
        sigma = random.uniform(12, 16)    # HIGH   (~5 % of samples – rare)

    noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
    gray  = np.clip(gray + noise, 0, 255)

    # 5. Normalise
    processed = gray / 255.0

    # 6. Star-visibility check (post-noise, pre-return)
    #    Reject patches where noise has buried the starfield signal:
    #      • mean too low  → patch is essentially empty sky / blank region
    #      • SNR too low   → bright pixels (stars) indistinguishable from noise
    #    SNR proxy: ratio of 95th-percentile brightness to noise std-dev.
    #    A healthy starfield patch has a few pixels noticeably above the floor.
    mean_val = float(np.mean(processed))
    if mean_val < (MEAN_LOW / 255.0):          # reuse global lower bound
        return None

    noise_std  = float(np.std(processed))
    peak_val   = float(np.percentile(processed, 95))
    snr_proxy  = (peak_val - mean_val) / (noise_std + 1e-6)
    if snr_proxy < 0.5:                        # stars buried in noise
        return None

    return processed.astype(np.float32)


def _add_sky_gradient(img: np.ndarray) -> np.ndarray:
    """
    Add a gentle smooth gradient across the patch to simulate the varying sky
    background brightness common in real astronomical frames.
    """
    h, w = img.shape
    angle  = random.uniform(0, 2 * math.pi)
    amp    = random.uniform(5, 30)        # peak-to-peak amplitude in DN

    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(xs, ys)

    gradient = amp * (math.cos(angle) * xx + math.sin(angle) * yy)
    return img + gradient.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STREAK GENERATION – physically realistic
# ══════════════════════════════════════════════════════════════════════════════
def generate_mask(
    size: int = PATCH_SIZE,
    min_length_frac: float = 0.30,
    max_length_frac: float = 1.20,
) -> np.ndarray:
    """
    Create a binary float32 mask (0 or 1) for a single satellite streak.

    Features
    --------
    • Straight line at a random angle
    • Thin width: 1–2 px
    • Variable length (can cross entire frame)
    • Random start position / angle
    • Variable brightness along the streak  (modulated sinusoid + noise)
    • Dashed / broken sections to mimic tumbling satellites
    • Soft PSF via Gaussian blur

    Returns
    -------
    mask : np.ndarray  shape (size, size), dtype float32, values in [0, 1]
    """
    mask = np.zeros((size, size), dtype=np.float32)

    # ── Geometry ──────────────────────────────────────────────
    angle_deg = random.uniform(0, 180)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    max_len  = int(size * max_length_frac)
    min_len  = int(size * min_length_frac)
    length   = random.randint(min_len, max_len)

    # Start point (allow streak to originate slightly outside the frame)
    margin = size // 4
    x0 = random.randint(-margin, size + margin)
    y0 = random.randint(-margin, size + margin)

    # ── Brightness modulation ─────────────────────────────────
    dashed       = random.random() < 0.40         # 40 % chance of dashed streak
    dash_period  = random.randint(8, 20)
    dash_duty    = random.uniform(0.4, 0.8)       # fraction of period that is "on"

    brightness_variation = random.random() < 0.70
    flicker_freq  = random.uniform(0.01, 0.05)    # cycles per pixel
    flicker_amp   = random.uniform(0.0, 0.5)

    # ── Draw pixel by pixel ───────────────────────────────────
    for t in range(length):
        px = int(x0 + t * cos_a)
        py = int(y0 + t * sin_a)

        if not (0 <= px < size and 0 <= py < size):
            continue

        # Dashed gap
        if dashed and (t % dash_period) / dash_period > dash_duty:
            continue

        # Per-pixel intensity  (range 0.5 – 1.0 before global brightness)
        intensity = 1.0
        if brightness_variation:
            intensity = 0.5 + 0.5 * (1.0 + math.sin(2 * math.pi * flicker_freq * t))
            intensity += random.uniform(-flicker_amp, flicker_amp)
            intensity = float(np.clip(intensity, 0.0, 1.0))

        # Optionally draw 2-px wide streak
        width = random.choice([1, 1, 2])     # mostly 1px, occasionally 2px
        for dw in range(width):
            npx = px
            npy = py + dw      # perpendicular offset (simplified)
            if 0 <= npx < size and 0 <= npy < size:
                mask[npy, npx] = max(mask[npy, npx], intensity)

    # ── Soft PSF blur (simulate atmospheric / optical spreading) ──
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Re-binarise at a low threshold so the mask stays close to the line
    # but keep the soft edge for realistic edge supervision
    # (values remain in [0, 1] for float mask; hard mask is derived at inference)
    return mask.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 8 + 9. ADD STREAK TO CLEAN IMAGE
# ══════════════════════════════════════════════════════════════════════════════
def add_streak(
    clean_img: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite the satellite streak onto the clean normalised image.

    Parameters
    ----------
    clean_img : float32 array in [0, 1]
    mask      : float32 array in [0, 1] (soft streak mask from generate_mask)

    Returns
    -------
    streak_img  : float32 [0, 1] – image with streak added
    binary_mask : uint8  {0, 255} – hard binary mask for training
    """
    brightness = random.uniform(0.3, 1.0)

    streak_img = clean_img + mask * brightness
    streak_img = np.clip(streak_img, 0.0, 1.0).astype(np.float32)

    # Hard binary mask: pixels where soft mask > 0.05 are considered "streak"
    binary_mask = (mask > 0.05).astype(np.uint8) * 255

    return streak_img, binary_mask


# ══════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("Could not read image: %s", path)
    return img


def save_float_image(arr: np.ndarray, path: str) -> None:
    """Save a float32 [0,1] image as 16-bit PNG for maximum fidelity."""
    img_16 = (arr * 65535).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(path, img_16)


def save_mask(mask: np.ndarray, path: str) -> None:
    cv2.imwrite(path, mask)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(
    input_dir: str,
    output_dir: str,
    target_samples: int = 2000,
    patch_size: int = PATCH_SIZE,
    seed: int = 42,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    for sub in ("images", "masks", "clean_images"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ✅ use sparse_stars folder
    src_dir = Path(input_dir) / "sparse_stars"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    src_files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]

    if not src_files:
        log.error("No images found in %s", src_dir)
        return

    log.info("Found %d source images in '%s'.", len(src_files), src_dir)
    log.info("Target (NEW samples): %d  |  Output: '%s'", target_samples, output_dir)

    base_per_image = target_samples // len(src_files)
    extra = target_samples % len(src_files)

    # ✅ continue numbering
    existing_files = list((out / "images").glob("*.png"))
    sample_idx = len(existing_files)
    start_idx = sample_idx   # 🔥 IMPORTANT

    pbar = tqdm(total=target_samples, desc="Generating", unit="sample")

    for file_idx, src_path in enumerate(src_files):
        n_patches_wanted = base_per_image + (1 if file_idx < extra else 0)

        img = load_image(str(src_path))
        if img is None:
            continue

        cropped = crop_image(img)
        if cropped is None:
            log.warning("Skipping %s – too small after crop.", src_path.name)
            continue

        n_request = min(n_patches_wanted * 3, max(n_patches_wanted, PATCHES_PER_IMAGE[1]))
        raw_patches = extract_patches(cropped, n_patches=n_request, patch_size=patch_size)

        generated_from_this = 0

        for patch in raw_patches:
            if generated_from_this >= n_patches_wanted:
                break

            clean = preprocess_patch(patch)
            if clean is None:
                continue

            mean_clean = clean.mean() * 255
            if mean_clean > MEAN_HIGH:
                continue

            fname = f"{sample_idx:05d}.png"

            save_float_image(clean, str(out / "clean_images" / fname))

            mask = generate_mask(size=patch_size)
            streak_img, binary_mask = add_streak(clean, mask)

            save_float_image(streak_img, str(out / "images" / fname))
            save_mask(binary_mask, str(out / "masks" / fname))

            sample_idx += 1
            generated_from_this += 1
            pbar.update(1)

            # 🔥 FIXED CONDITION
            if (sample_idx - start_idx) >= target_samples:
                break

        # 🔥 FIXED CONDITION
        if (sample_idx - start_idx) >= target_samples:
            break

    pbar.close()

    log.info("Done. Generated %d NEW samples.", sample_idx - start_idx)
    log.info("Total dataset size: %d", sample_idx)

    _print_summary(out, sample_idx)


def _print_summary(out: Path, n: int) -> None:
    print("\n" + "═" * 54)
    print("  Dataset generation complete")
    print("═" * 54)
    print(f"  Total samples  : {n}")
    print(f"  images/        : {out / 'images'}")
    print(f"  masks/         : {out / 'masks'}")
    print(f"  clean_images/  : {out / 'clean_images'}")
    print("═" * 54 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic satellite-streak dataset generator for U-Net training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",  "-i",
        required=True,
        help="Folder containing raw starfield images (JPG / PNG / TIFF).",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="dataset",
        help="Root output directory; sub-folders are created automatically.",
    )
    parser.add_argument(
        "--target",     "-n",
        type=int,
        default=2000,
        help="Total number of (image, mask, clean) triplets to generate.",
    )
    parser.add_argument(
        "--patch_size", "-p",
        type=int,
        default=256,
        help="Square patch size in pixels.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        input_dir      = args.input_dir,
        output_dir     = args.output_dir,
        target_samples = args.target,
        patch_size     = args.patch_size,
        seed           = args.seed,
    )
