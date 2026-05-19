import torch
import numpy as np
from pathlib import Path
from PIL import Image
import random
import cv2

try:
    import segmentation_models_pytorch as smp
except ImportError:
    pass

from model import UNet

DEVICE = torch.device("cpu")

def analyze_mask(pred, target):
    # Flatten arrays
    p = pred.flatten().astype(bool)
    t = target.flatten().astype(bool)
    
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    tn = np.sum(~p & ~t)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    gt_pixels = np.sum(t)
    pred_pixels = np.sum(p)
    
    return tp, fp, fn, tn, precision, recall, gt_pixels, pred_pixels

def run_analysis():
    print("Loading models...")
    
    old_model = UNet(in_channels=1, out_channels=1)
    try:
        old_model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
        old_model.eval()
    except:
        print("Could not load best_unet.pth")
        return

    new_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1, activation=None)
    try:
        new_model.load_state_dict(torch.load("best_resnet34_unet.pth", map_location=DEVICE))
        new_model.eval()
    except:
        print("Could not load best_resnet34_unet.pth")
        return

    image_dir = Path("dataset/images")
    mask_dir = Path("dataset/masks")
    
    image_paths = sorted(list(image_dir.glob("*.png")))
    random.seed(10)
    sample_paths = random.sample(image_paths, min(100, len(image_paths)))
    
    stats = {
        "old": {"tp":0, "fp":0, "fn":0, "gt_area":0, "pred_area":0, "precisions":[], "recalls":[]},
        "new": {"tp":0, "fp":0, "fn":0, "gt_area":0, "pred_area":0, "precisions":[], "recalls":[]}
    }
    
    for i, img_path in enumerate(sample_paths):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists(): continue
            
        pil_img = Image.open(img_path)
        if pil_img.mode in ("I;16", "I"):
            img_f32 = np.array(pil_img, dtype=np.float32) / 65535.0
        else:
            img_f32 = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
        mask_gt = np.array(Image.open(mask_path).convert("L")) > 127
        
        tensor = torch.from_numpy(img_f32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            old_logit = old_model(tensor)
            new_logit = new_model(tensor)
            
        old_pred = (torch.sigmoid(old_logit).squeeze().numpy() > 0.5)
        # Using 0.5 threshold for fair diagnostic comparison, can be adjusted
        new_pred = (torch.sigmoid(new_logit).squeeze().numpy() > 0.5)
        
        # Analyze old
        tp, fp, fn, tn, prec, rec, gt_a, pr_a = analyze_mask(old_pred, mask_gt)
        stats["old"]["tp"] += tp
        stats["old"]["fp"] += fp
        stats["old"]["fn"] += fn
        stats["old"]["gt_area"] += gt_a
        stats["old"]["pred_area"] += pr_a
        stats["old"]["precisions"].append(prec)
        stats["old"]["recalls"].append(rec)
        
        # Analyze new
        tp, fp, fn, tn, prec, rec, gt_a, pr_a = analyze_mask(new_pred, mask_gt)
        stats["new"]["tp"] += tp
        stats["new"]["fp"] += fp
        stats["new"]["fn"] += fn
        stats["new"]["gt_area"] += gt_a
        stats["new"]["pred_area"] += pr_a
        stats["new"]["precisions"].append(prec)
        stats["new"]["recalls"].append(rec)

    for name in ["old", "new"]:
        print(f"\n=== {name.upper()} MODEL DIAGNOSTICS (100 Samples) ===")
        avg_prec = np.mean(stats[name]["precisions"]) * 100
        avg_rec = np.mean(stats[name]["recalls"]) * 100
        total_tp = stats[name]["tp"]
        total_fp = stats[name]["fp"]
        total_fn = stats[name]["fn"]
        
        print(f"Total True Positives  : {total_tp}")
        print(f"Total False Positives : {total_fp} (Pixels hallucinated/too thick)")
        print(f"Total False Negatives : {total_fn} (Pixels missed/faint streaks lost)")
        print(f"Average Precision     : {avg_prec:.2f}% (If it predicts a streak, is it real?)")
        print(f"Average Recall        : {avg_rec:.2f}% (Did it find all the real streak pixels?)")
        print(f"Avg GT Area / image   : {stats[name]['gt_area'] / 100:.1f} pixels")
        print(f"Avg Pred Area / image : {stats[name]['pred_area'] / 100:.1f} pixels")

if __name__ == "__main__":
    run_analysis()
