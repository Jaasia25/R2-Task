from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import base64
import os
import sys
import torch

# ⬇️ Import your GradCAM class
from xai.gradcam import yolo_heatmap, getParams, letterbox
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import getUtils


app = FastAPI()
utils = getUtils()
config = utils.load_yaml()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# LOAD YOLO MODEL
# ------------------------------------------------------
YOLO_MODEL_PATH = config['model_path']['MODEL']
yolo_model = YOLO(YOLO_MODEL_PATH)

# ------------------------------------------------------
# LOAD EXPLAINABLE AI MODEL
# ------------------------------------------------------
params = getParams(YOLO_MODEL_PATH).get_params()
gradcam_model = yolo_heatmap(**params)


# 🔥 Utility: Numpy/BGR → base64
def to_base64(img):
    """Convert numpy image to base64 safely"""
    if img is None or img.size == 0:
        raise ValueError("❌ to_base64() received an empty image")

    # If image is RGB, convert to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("❌ Failed to encode image to JPG")

    return base64.b64encode(buffer).decode("utf-8")


@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    # Store original dimensions
    original_height, original_width = img_np.shape[:2]

    # ------------------------------------------------------
    # PREPROCESS IMAGE SAME WAY AS GRADCAM
    # ------------------------------------------------------
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Apply letterboxing (same as GradCAM)
    img_letterboxed, ratio, (top, bottom, left, right) = letterbox(
        img_bgr,
        new_shape=(640, 640),
        auto=True
    )
    
    # Convert to RGB and normalize for tensor
    img_letterboxed_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_letterboxed_rgb) / 255.0
    
    # Create tensor
    tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0)
    
    # ------------------------------------------------------
    # 1️⃣ DETECTION (using letterboxed image)
    # ------------------------------------------------------
    CONF_THRESHOLD = 0.2  # Same as GradCAM default
    
    results = yolo_model.predict(tensor, save=False, conf=CONF_THRESHOLD, iou=0.7)
    result = results[0]

    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    names = result.names

    # ------------------------------------------------------
    # COUNT SPECIES
    # ------------------------------------------------------
    species_count = {}
    for cid in class_ids:
        label = names.get(cid, "Unknown")
        species_count[label] = species_count.get(label, 0) + 1

    # ------------------------------------------------------
    # CREATE ANNOTATED IMAGE (on letterboxed image)
    # ------------------------------------------------------
    # Use result.plot with the ORIGINAL letterboxed BGR image (uint8, 0-255)
    annotated_img_letterboxed = result.plot(
        img=img_letterboxed.copy(),  # BGR uint8 image
        conf=True,
        line_width=2,
        font_size=1.0,
        labels=True,
        boxes=True
    )
    
    # Convert BGR to RGB
    annotated_img_letterboxed = cv2.cvtColor(annotated_img_letterboxed, cv2.COLOR_BGR2RGB)
    
    # Remove letterbox padding to get back to original aspect ratio
    annotated_img = annotated_img_letterboxed[
        top:annotated_img_letterboxed.shape[0] - bottom,
        left:annotated_img_letterboxed.shape[1] - right
    ]
    
    # Resize back to original dimensions
    annotated_img = cv2.resize(annotated_img, (original_width, original_height))

    # ------------------------------------------------------
    # 2️⃣ EXPLAINABLE AI (GradCAM) - using same preprocessing
    # ------------------------------------------------------
    heatmap = gradcam_model.run_from_array(img_np)

    if heatmap is None:
        return {"error": "GradCAM returned None — fix run_from_array()"}

    # Resize heatmap to match original dimensions (for consistency)
    heatmap = cv2.resize(heatmap, (original_width, original_height))

    # ------------------------------------------------------
    # GET MAX CONFIDENCE
    # ------------------------------------------------------
    max_conf = None
    if boxes.conf.numel() > 0:
        max_conf = float(boxes.conf.max().item())

    # ------------------------------------------------------
    # RETURN EVERYTHING AS JSON
    # ------------------------------------------------------
    return {
        "total_fish": len(class_ids),
        "species_count": species_count,
        "annotated_image": to_base64(annotated_img),
        "heatmap_image": to_base64(heatmap),
        "original_image": to_base64(img_np),
        "gradcam_score": max_conf
    }