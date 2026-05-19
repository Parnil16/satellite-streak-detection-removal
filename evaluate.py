import torch
import numpy as np
from pathlib import Path
from PIL import Image
from model import UNet
import random

DEVICE = torch.device("cpu")

def compute_iou_dice(pred, target):
    intersection = np.sum(np.logical_and(target, pred))
    union = np.sum(np.logical_or(target, pred))
    
    iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
    
    dice_denom = np.sum(pred) + np.sum(target)
    dice = (2.0 * intersection) / dice_denom if dice_denom > 0 else (1.0 if intersection == 0 else 0.0)
    
    return iou, dice

def main():
    model = UNet(in_channels=1, out_channels=1)
    state_dict = torch.load("best_unet.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    image_dir = Path("dataset/images")
    mask_dir = Path("dataset/masks")
    
    image_paths = sorted(list(image_dir.glob("*.png")))
    random.seed(42)
    sample_paths = random.sample(image_paths, min(200, len(image_paths)))
    
    ious = []
    dices = []
    
    for i, img_path in enumerate(sample_paths):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
            
        pil_img = Image.open(img_path)
        if pil_img.mode in ("I;16", "I"):
            img_f32 = np.array(pil_img, dtype=np.float32) / 65535.0
        else:
            img_f32 = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
        
        pil_mask = Image.open(mask_path).convert("L")
        mask_gt = np.array(pil_mask) > 127
        
        tensor = torch.from_numpy(img_f32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).squeeze().numpy()
            
        pred = prob > 0.5
        
        iou, dice = compute_iou_dice(pred, mask_gt)
        ious.append(iou)
        dices.append(dice)
        
    if len(ious) > 0:
        print(f"Evaluated {len(ious)} images.")
        print(f"Mean IoU: {np.mean(ious)*100:.2f}%")
        print(f"Mean Dice Score: {np.mean(dices)*100:.2f}%")

if __name__ == '__main__':
    main()
