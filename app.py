import streamlit as st
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
from PIL import Image
import tempfile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Set page config
st.set_page_config(page_title="Road Segmentation App", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.load_state_dict(torch.load('./model/segmentation_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Function to process a single frame/image
def process_image(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out']
        prob_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (prob_mask > 0.7).astype(np.uint8)

    # Resize back to original
    mask = (pred_mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    return mask_bgr

# Streamlit UI
st.title("🧠 Road Segmentation with DeepLabV3")
file = st.file_uploader("📂 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if file is not None:
    file_bytes = file.read()

    if file.type.startswith("image"):
        image = np.array(Image.open(file).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = process_image(image_bgr)
        side_by_side = np.hstack((image_bgr, mask))

        st.image(side_by_side, channels="BGR", caption="Original (Left) | Mask (Right)")

    elif file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file_bytes)
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(output_temp.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        stframe = st.empty()
        st.info("Processing video. This may take a moment...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask = process_image(frame)
            blended = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
            out.write(blended)

            stframe.image(np.hstack((frame, mask)), channels="BGR", caption="Original | Mask")

        cap.release()
        out.release()

        with open(output_temp.name, 'rb') as f:
            st.success("✅ Video processed. Download below:")
            st.download_button(
                label="📥 Download Segmented Video",
                data=f,
                file_name="segmented_output.mp4",
                mime="video/mp4"
            )
