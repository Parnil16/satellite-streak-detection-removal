"""
main.py
Satellite Streak Detection & Inpainting Pipeline
CPU-only | Local execution | VS Code compatible

Usage:
    python main.py --image path/to/image.png
    python main.py --image path/to/image.png --threshold 0.4 --save
"""

import sys
import argparse
import math
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("\n[ERROR] Missing new dependencies for the ResNet34 model!")
    print("Please run this on your local computer:")
    print("  pip install segmentation-models-pytorch albumentations\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH  = "best_resnet34_unet.pth"   # new checkpoint name
DEVICE      = torch.device("cpu")
THRESHOLD   = 0.4
DILATE_K    = 3
DILATE_ITER = 1
INPAINT_R   = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH) -> torch.nn.Module:
    """
    Recreate U-Net, load saved weights, and set to eval mode.
    Performs a key-name sanity check before loading to give clear
    error messages if the checkpoint doesn't match this architecture.
    """
    path = Path(model_path)

    # ── File existence ────────────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found: {path.resolve()}\n"
            f"  → Copy 'best_unet.pth' into the project root:\n"
            f"     {Path('.').resolve()}"
        )

    # ── Load raw state-dict (CPU only — no CUDA required) ────────────────────
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
    except Exception as exc:
        raise RuntimeError(
            f"\n[ERROR] Failed to read checkpoint: {exc}\n"
            f"  → The file may be corrupt or saved with an incompatible "
            f"PyTorch version."
        ) from exc

    # ── Architecture / key mismatch check ────────────────────────────────────
    # ── Architecture / key mismatch check ────────────────────────────────────
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    )
    
    model_keys   = set(model.state_dict().keys())
    ckpt_keys    = set(state_dict.keys())

    missing  = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        msg = "\n[ERROR] Checkpoint keys do not match the ResNet34 architecture.\n"
        if missing:
            sample = list(missing)[:5]
            msg += f"  Missing  ({len(missing)} keys, e.g.): {sample}\n"
        if unexpected:
            sample = list(unexpected)[:5]
            msg += f"  Unexpected ({len(unexpected)} keys, e.g.): {sample}\n"
        msg += (
            "  → Make sure you downloaded the correct 'best_resnet34_unet.pth'."
        )
        raise RuntimeError(msg)

    # ── Load weights ──────────────────────────────────────────────────────────
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded  |  {n_params:,} parameters  |  device: {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """
    Load a grayscale PNG (8-bit or 16-bit).

    Returns
    -------
    img_f32 : (H, W) float32 in [0, 1]  — model input
    img_u8  : (H, W) uint8  in [0, 255] — OpenCV inpainting input
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Image not found: {path.resolve()}"
        )

    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits"):
        print(f"[WARN] Unusual file extension '{path.suffix}' — attempting to load anyway.")

    try:
        pil_img = Image.open(path)
    except Exception as exc:
        raise RuntimeError(f"\n[ERROR] Cannot open image: {exc}") from exc

    if pil_img.mode in ("I;16", "I"):           # 16-bit grayscale
        img_f32 = np.array(pil_img, dtype=np.float32) / 65535.0
    else:                                        # 8-bit (L, RGB, RGBA …)
        img_f32 = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0

    img_u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)

    print(f"[OK] Image loaded  |  {img_f32.shape[1]}×{img_f32.shape[0]} px  "
          f"|  mode: {pil_img.mode}")
    return img_f32, img_u8


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_mask(
    model:     torch.nn.Module,
    img_f32:   np.ndarray,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """
    Run U-Net inference and return a binary uint8 mask {0, 255}.

    Parameters
    ----------
    model     : loaded U-Net in eval mode
    img_f32   : (H, W) float32 image in [0, 1]
    threshold : sigmoid threshold for binarisation
    """
    # (H,W) → (1,1,H,W) tensor
    tensor = (
        torch.from_numpy(img_f32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)         # CPU tensor — no .cuda() call
    )

    with torch.no_grad():
        logit = model(tensor)                               # (1,1,H,W)
        prob  = torch.sigmoid(logit).squeeze().numpy()      # (H,W) float32

    binary  = (prob > threshold).astype(np.uint8)          # {0, 1}
    mask_u8 = binary * 255                                 # {0, 255}

    streak_pct = 100.0 * binary.sum() / binary.size
    print(f"[OK] Mask predicted  |  threshold: {threshold}  "
          f"|  streak pixels: {binary.sum()} ({streak_pct:.2f}%)")
    return mask_u8


# ─────────────────────────────────────────────────────────────────────────────
# 4. Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def dilate_mask(
    mask_u8:    np.ndarray,
    kernel_size: int = DILATE_K,
    iterations:  int = DILATE_ITER,
) -> np.ndarray:
    """
    Morphological dilation to ensure full streak pixel coverage.
    Returns uint8 mask {0, 255}.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=iterations)
    print(f"[OK] Mask dilated   |  kernel: {kernel_size}×{kernel_size}  "
          f"|  iterations: {iterations}")
    return dilated


# ─────────────────────────────────────────────────────────────────────────────
# 5. Overlay
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay(
    img_u8:  np.ndarray,
    mask_u8: np.ndarray,
    alpha:   float = 0.45,
) -> np.ndarray:
    """
    Blend a red streak highlight over the grayscale image.

    Returns
    -------
    overlay : (H, W, 3) uint8 RGB image
    """
    rgb        = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    red_layer  = np.zeros_like(rgb)
    red_layer[:, :, 0] = mask_u8

    streak_px = mask_u8 > 0
    overlay   = rgb.copy()
    overlay[streak_px] = (
        (1.0 - alpha) * rgb[streak_px].astype(np.float32) +
        alpha         * red_layer[streak_px].astype(np.float32)
    ).astype(np.uint8)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# 6. Streak Subtraction
# ─────────────────────────────────────────────────────────────────────────────

def estimate_background(img_f32: np.ndarray, mask_u8: np.ndarray, box_size: int = 32) -> np.ndarray:
    H, W = img_f32.shape
    grid_y = np.arange(box_size // 2, H, box_size)
    grid_x = np.arange(box_size // 2, W, box_size)
    
    bg_grid = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    
    for iy, cy in enumerate(grid_y):
        for ix, cx in enumerate(grid_x):
            y0 = max(0, cy - box_size // 2)
            y1 = min(H, cy + box_size // 2)
            x0 = max(0, cx - box_size // 2)
            x1 = min(W, cx + box_size // 2)
            
            box_img  = img_f32[y0:y1, x0:x1]
            box_mask = mask_u8[y0:y1, x0:x1]
            
            unmasked = box_img[box_mask == 0]
            if len(unmasked) > 4:
                median = np.median(unmasked)
                std = unmasked.std()
                clipped = unmasked[np.abs(unmasked - median) < 2.5 * std]
                bg_grid[iy, ix] = np.median(clipped) if len(clipped) > 0 else median
            else:
                bg_grid[iy, ix] = np.nan
                
    nan_mask = np.isnan(bg_grid)
    if nan_mask.any():
        indices = ndimage.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        bg_grid = bg_grid[tuple(indices)]
        
    spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=min(3, len(grid_y)-1), ky=min(3, len(grid_x)-1))
    background = spline(np.arange(H), np.arange(W)).astype(np.float32)
    return background

def subtract_streak_median_reference(img_f32: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Astronomical streak subtraction using directional median filtering.
    """
    # 1. Background Estimation
    bg = estimate_background(img_f32, mask_u8, box_size=32)
    img_no_bg = img_f32 - bg
    
    # 2. Angle Estimation using robust cv2.fitLine
    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 10:
        print("[WARN] Mask is empty or too small, skipping subtraction.")
        return (img_f32 * 255).clip(0, 255).astype(np.uint8)
        
    # 2. Angle Estimation
    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 10:
        print("[WARN] Mask is empty or too small, skipping subtraction.")
        return (img_f32 * 255).clip(0, 255).astype(np.uint8)
        
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    angle_rad = math.atan2(vy[0], vx[0])
    angle_deg = math.degrees(angle_rad)
    
    # Dynamic filter size based on streak length
    x, y, w, h = cv2.boundingRect(pts)
    streak_length = math.sqrt(w**2 + h**2)
    filter_size = max(15, min(int(streak_length // 3), 101))
    if filter_size % 2 == 0: filter_size += 1
    
    # 3. Rotate to horizontal (float32, INTER_CUBIC)
    H, W = img_f32.shape
    center = (W // 2, H // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    rotated_img = cv2.warpAffine(img_no_bg, M_rot, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    rotated_mask = cv2.warpAffine(mask_u8, M_rot, (W, H), flags=cv2.INTER_NEAREST)
    
    # 4. Directional Median Filter
    kernel = np.ones((1, filter_size), dtype=np.uint8)
    streak_model_rot = ndimage.median_filter(rotated_img, footprint=kernel)
    
    # Keep only the streak inside the mask to prevent subtracting median sky
    streak_model_rot[rotated_mask == 0] = 0.0
    
    # 5. Rotate back
    M_inv = cv2.invertAffineTransform(M_rot)
    streak_model = cv2.warpAffine(streak_model_rot, M_inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    
    # 6. Subtract (Masked)
    streak_pixels = mask_u8 > 0
    result_f32 = img_f32.copy()
    result_f32[streak_pixels] = img_f32[streak_pixels] - streak_model[streak_pixels]
    
    # 7. Photometric Validation
    unmasked = img_f32[mask_u8 == 0]
    unmasked_std = unmasked.std() if len(unmasked) > 0 else 0
    
    residual = result_f32[streak_pixels]
    bg_residual = bg[streak_pixels]
    residual_noise = (residual - bg_residual).std() if len(residual) > 0 else 0
    
    print(f"[OK] Streak Subtracted | angle: {angle_deg:.1f}° | filter_size: {filter_size}")
    print(f"     -> Background noise: {unmasked_std:.4f} | Residual noise: {residual_noise:.4f}")
    
    return (result_f32 * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
def _robust_std(values: np.ndarray) -> float:
    """Median absolute deviation estimate of Gaussian sigma."""
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    """Fill sparse 1D tracks by interpolation."""
    values = values.astype(np.float32, copy=True)
    valid = np.isfinite(values)
    if not valid.any():
        values[:] = 0.0
        return values

    idx = np.arange(values.size)
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def _smooth_active_runs(values: np.ndarray, active: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth only inside active streak runs so dashed gaps stay dark."""
    out = np.zeros_like(values, dtype=np.float32)
    active = active.astype(bool)
    start = None

    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        at_end = i == active.size - 1
        if start is not None and ((not is_active) or at_end):
            stop = i + 1 if is_active and at_end else i
            run = values[start:stop]
            if run.size <= 2:
                out[start:stop] = run
            else:
                out[start:stop] = ndimage.gaussian_filter1d(run, sigma=sigma, mode="nearest")
            start = None

    return out


def subtract_streak(img_f32: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Photometric streak subtraction with a gap-aware directional model.

    The directional median model is retained because it is strong for
    continuous trails.  A separate along-track activity gate is estimated from
    the background-subtracted image and applied column-by-column after
    rotation, so dashed gaps are not filled by the long median footprint.
    """
    bg = estimate_background(img_f32, mask_u8, box_size=32)
    img_no_bg = img_f32 - bg

    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 10:
        print("[WARN] Mask is empty or too small, skipping subtraction.")
        return (img_f32 * 255).clip(0, 255).astype(np.uint8)

    [vx, vy, _x0, _y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    angle_rad = math.atan2(vy[0], vx[0])
    angle_deg = math.degrees(angle_rad)

    H, W = img_f32.shape
    center = (W // 2, H // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated_img = cv2.warpAffine(
        img_no_bg, M_rot, (W, H), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    rotated_mask = cv2.warpAffine(
        mask_u8, M_rot, (W, H), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    mask_bool = rotated_mask > 0
    ys_all, xs_all = np.where(mask_bool)
    if xs_all.size < 10:
        print("[WARN] Rotated mask is empty or too small, skipping subtraction.")
        return (img_f32 * 255).clip(0, 255).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(pts)
    streak_length = math.sqrt(w**2 + h**2)
    filter_size = max(15, min(int(streak_length // 3), 101))
    if filter_size % 2 == 0:
        filter_size += 1

    col_counts = np.bincount(xs_all, minlength=W).astype(np.float32)
    sky_noise = _robust_std(rotated_img[~mask_bool])
    if sky_noise <= 0:
        sky_noise = float(rotated_img[~mask_bool].std()) if np.any(~mask_bool) else 0.0

    col_signal = np.zeros(W, dtype=np.float32)
    for x in np.unique(xs_all):
        vals = rotated_img[mask_bool[:, x], x]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            col_signal[x] = max(0.0, float(np.percentile(vals, 90)))

    col_signal = ndimage.gaussian_filter1d(col_signal, sigma=1.0, mode="nearest")
    active_threshold = max(1.25 * sky_noise, 0.006)
    active = (col_counts > 0) & (col_signal > active_threshold)

    # Heal tiny detector holes in continuous trails without bridging real dash gaps.
    active = ndimage.binary_closing(active, structure=np.ones(3, dtype=bool))
    active = ndimage.binary_opening(active, structure=np.ones(2, dtype=bool))

    kernel = np.ones((1, filter_size), dtype=np.uint8)
    streak_model_rot = ndimage.median_filter(rotated_img, footprint=kernel)
    streak_model_rot[rotated_mask == 0] = 0.0
    streak_model_rot[:, ~active] = 0.0

    positive_limit = np.maximum(0.0, rotated_img + 2.5 * sky_noise)
    streak_model_rot = np.minimum(np.maximum(streak_model_rot, 0.0), positive_limit)

    M_inv = cv2.invertAffineTransform(M_rot)
    streak_model = cv2.warpAffine(
        streak_model_rot, M_inv, (W, H), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    streak_model = np.maximum(streak_model, 0.0)

    streak_pixels = mask_u8 > 0
    result_f32 = img_f32.copy()
    result_f32[streak_pixels] = img_f32[streak_pixels] - streak_model[streak_pixels]

    unmasked = img_f32[mask_u8 == 0]
    unmasked_std = _robust_std(unmasked) if len(unmasked) > 0 else 0.0
    residual = result_f32[streak_pixels]
    bg_residual = bg[streak_pixels]
    residual_noise = _robust_std(residual - bg_residual) if len(residual) > 0 else 0.0

    active_pct = 100.0 * active.sum() / max(1, int(xs_all.max()) - int(xs_all.min()) + 1)
    print(
        f"[OK] Streak Subtracted | angle: {angle_deg:.1f} deg | "
        f"filter_size: {filter_size} | active columns: {active.sum()} ({active_pct:.1f}%)"
    )
    print(f"     -> Background noise: {unmasked_std:.4f} | Residual noise: {residual_noise:.4f}")

    return (result_f32 * 255).clip(0, 255).astype(np.uint8)


# 7. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise(results: dict, image_name: str = "", save_path: str = None):
    """
    Show original / mask / overlay / inpainted in a 1×4 grid.

    Parameters
    ----------
    results    : dict returned by run_pipeline()
    image_name : used in the figure title
    save_path  : if given, figure is saved here before display
    """
    panels = [
        ("Original Image",  results["original"],  "gray"),
        ("Predicted Mask",  results["mask"],       "gray"),
        ("Streak Overlay",  results["overlay"],    None),
        ("Inpainted Image", results["inpainted"],  "gray"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Satellite Streak Pipeline  —  {image_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Figure saved  →  {save_path}")

    plt.show()


def save_outputs(results: dict, out_dir: str = "outputs"):
    """
    Write each pipeline result to its own PNG file.
    The overlay is converted BGR before writing (OpenCV convention).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "original.png"),  results["original"])
    cv2.imwrite(str(out / "mask.png"),      results["mask"])
    cv2.imwrite(str(out / "inpainted.png"), results["inpainted"])
    cv2.imwrite(str(out / "overlay.png"),
                cv2.cvtColor(results["overlay"], cv2.COLOR_RGB2BGR))

    print(f"[OK] Files saved to: {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path:  str,
    model:       torch.nn.Module,
    threshold:   float = THRESHOLD,
    dilate_k:    int   = DILATE_K,
    dilate_iter: int   = DILATE_ITER,
    inpaint_r:   int   = INPAINT_R,
) -> dict:
    """
    End-to-end streak detection and removal.

    Returns
    -------
    dict with keys:
        'original'  : (H, W)    uint8
        'mask'      : (H, W)    uint8  {0, 255}
        'overlay'   : (H, W, 3) uint8  RGB
        'inpainted' : (H, W)    uint8
    """
    print(f"\n{'─' * 52}")
    print(f"  Image : {image_path}")
    print(f"{'─' * 52}")

    img_f32, img_u8 = load_image(image_path)
    raw_mask        = predict_mask(model, img_f32, threshold)
    mask            = dilate_mask(raw_mask, dilate_k, dilate_iter)
    overlay         = make_overlay(img_u8, mask)
    inpainted       = subtract_streak(img_f32, mask)
    
    print(f"{'─' * 52}\n")

    return {
        "original" : img_u8,
        "mask"     : mask,
        "overlay"  : overlay,
        "inpainted": inpainted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Satellite streak detection and inpainting (CPU, local)"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input grayscale PNG image"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to trained U-Net checkpoint (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Sigmoid threshold for mask binarisation (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--dilate-k", type=int, default=DILATE_K,
        help=f"Dilation kernel size (default: {DILATE_K})"
    )
    parser.add_argument(
        "--dilate-iter", type=int, default=DILATE_ITER,
        help=f"Dilation iterations (default: {DILATE_ITER})"
    )
    parser.add_argument(
        "--inpaint-r", type=int, default=INPAINT_R,
        help=f"Inpainting radius in pixels (default: {INPAINT_R})"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save individual PNGs to ./outputs/"
    )
    parser.add_argument(
        "--save-figure", default=None,
        help="Path to save the summary figure (e.g. result.png)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model once — reuse for all images
    try:
        model = load_model(args.model)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(
            image_path  = args.image,
            model       = model,
            threshold   = args.threshold,
            dilate_k    = args.dilate_k,
            dilate_iter = args.dilate_iter,
            inpaint_r   = args.inpaint_r,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        sys.exit(1)

    # Visualise
    visualise(
        results,
        image_name = Path(args.image).name,
        save_path  = args.save_figure,
    )

    # Optionally save individual files
    if args.save:
        save_outputs(results)


if __name__ == "__main__":
    main()
