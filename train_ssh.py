import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import time

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("Please run: pip install segmentation-models-pytorch albumentations opencv-python")

import albumentations as A
from albumentations.pytorch import ToTensorV2

class StreakDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.image_dir.glob("*.png")))
        self.augment = augment
        
        # Albumentations transforms (expects uint8 images [0, 255] by default)
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        pil_img = Image.open(img_path)
        if pil_img.mode in ("I;16", "I"):
            image = np.array(pil_img, dtype=np.float32) / 65535.0
        else:
            image = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
            
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        mask = mask.astype(np.float32)

        # Albumentations natively supports 2D arrays for grayscale
        transformed = self.transform(image=image, mask=mask)
        
        image = transformed['image'].float()
        # ToTensorV2 expands dims for image but not always for mask, ensure (1, H, W)
        mask = transformed['mask'].unsqueeze(0).float()
        
        return image, mask

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # Use a powerful pre-trained backbone via SMP
    model = smp.Unet(
        encoder_name="resnet34",        # ResNet34 backbone
        encoder_weights="imagenet",     # Use pre-trained weights
        in_channels=1,                  # Grayscale images natively supported by SMP
        classes=1,                      # Binary segmentation
        activation=None                 # Raw logits out (we use BCEWithLogitsLoss)
    ).to(DEVICE)

    # Focal Loss + Dice Loss combined
    criterion_focal = smp.losses.FocalLoss("binary")
    criterion_dice = smp.losses.DiceLoss("binary", from_logits=True)
    
    def hybrid_loss(preds, targets):
        # 70% Dice (structural), 30% Focal (pixel-wise for imbalance)
        return 0.7 * criterion_dice(preds, targets) + 0.3 * criterion_focal(preds, targets)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Try to find the dataset directory automatically
    if Path("dataset/images").exists() and len(list(Path("dataset/images").glob("*.png"))) > 0:
        img_dir, mask_dir = "dataset/images", "dataset/masks"
    elif Path("dataset/dataset/images").exists() and len(list(Path("dataset/dataset/images").glob("*.png"))) > 0:
        img_dir, mask_dir = "dataset/dataset/images", "dataset/dataset/masks"
    elif Path("dataset/upload_data/images").exists():
        img_dir, mask_dir = "dataset/upload_data/images", "dataset/upload_data/masks"
    else:
        raise FileNotFoundError("Could not find the dataset images! Check the dataset extraction path.")

    print(f"Loading dataset from: {img_dir}")
    dataset = StreakDataset(img_dir, mask_dir, augment=True)
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! Found 0 images in {img_dir}.")
        
    # Simple split (90% train, 10% val)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    epochs = 50
    best_iou = 0.0
    
    # Modern Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(x)
                loss = hybrid_loss(preds, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_ious = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(x)
                    loss = hybrid_loss(preds, y)
                val_loss += loss.item()
                
                # Strict Hard Per-Image Mean IoU Calculation
                preds_binary = (torch.sigmoid(preds) > 0.5).bool()
                y_binary = y.bool()
                
                for i in range(x.size(0)):
                    p = preds_binary[i]
                    t = y_binary[i]
                    inter = (p & t).sum().item()
                    union = (p | t).sum().item()
                    
                    if union == 0:
                        val_ious.append(1.0 if inter == 0 else 0.0)
                    else:
                        val_ious.append(inter / union)
                        
        mean_iou = np.mean(val_ious)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Mean Val IoU: {mean_iou*100:.2f}%")
        
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "best_resnet34_unet.pth")
            print("  -> Saved new best model!")

if __name__ == "__main__":
    train()
