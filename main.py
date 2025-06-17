import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # Binary mask output
model.load_state_dict(torch.load(r'.\model\segmentation_model.pth', map_location=device))
model.eval()
model.to(device)

# Albumentations transform (same as training)
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load video
cap = cv2.VideoCapture(r"C:\Users\HP\Desktop\Sharedfolder\cv_projects\data\video\5943714-hd_1920_1080_30fps.mp4")

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("✅ End of video")
        break

    # Convert BGR to RGB for model
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # Predict mask
    with torch.no_grad():
        output = model(input_tensor)['out']
        prob_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (prob_mask > 0.6).astype(np.uint8)

    # Resize mask back to original frame size
    mask_resized = cv2.resize(pred_mask * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # Combine original and mask side-by-side
    side_by_side = np.hstack((img, mask_bgr))

    frame_resized = cv2.resize(side_by_side, (1280, 720))

    # Display the frame
    cv2.imshow('Original (Left) | Mask (Right)', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
