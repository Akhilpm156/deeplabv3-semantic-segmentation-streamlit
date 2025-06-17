import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Custom Dataset Class
# -----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.jpg') and os.path.exists(os.path.join(folder_path, f.replace('.jpg', '_mask.png')))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace('.jpg', '_mask.png')

        image_path = os.path.join(self.folder_path, image_name)
        mask_path = os.path.join(self.folder_path, mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# -----------------------------
# Transformations
# -----------------------------
def get_transforms(train=True):
    base_transforms = [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2)
        ] + base_transforms)
    return A.Compose(base_transforms)

# -----------------------------
# Dice Score Metric
# -----------------------------
def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = 2 * intersection / (union + 1e-8)
    return dice.item()

# -----------------------------
# Model Setup
# -----------------------------
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # Binary output
model = model.to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Dataloaders
# -----------------------------
train_ds = SegmentationDataset("./data/train", transform=get_transforms(train=True))
val_ds = SegmentationDataset("./data/valid", transform=get_transforms(train=False))

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(2):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.unsqueeze(1).float().to(device)

        outputs = model(images)['out']
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.unsqueeze(1).float().to(device)
            outputs = model(images)['out']
            total_dice += dice_score(outputs, masks)

    print(f"Epoch {epoch+1}, Dice Score: {total_dice / len(val_loader):.4f}")

# -----------------------------
# Save the Model
# -----------------------------
os.makedirs('./model', exist_ok=True)
torch.save(model.state_dict(), './model/segmentation_model.pth')
