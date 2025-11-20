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

# ⬇️ Import your GradCAM class
from xai.gradcam import yolo_heatmap, get_params

app = FastAPI()

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
YOLO_MODEL_PATH = "models/fish_yolov8/weights/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# ------------------------------------------------------
# LOAD EXPLAINABLE AI MODEL
# ------------------------------------------------------
params = get_params()
gradcam_model = yolo_heatmap(**params)


# 🔥 Utility: Numpy/BGR → base64
def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    # ------------------------------------------------------
    # 1️⃣ DETECTION
    # ------------------------------------------------------
    results = yolo_model.predict(img_np, save=False, conf=0.30)
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
    # CREATE YOLO ANNOTATED IMAGE
    # ------------------------------------------------------
    annotated_img = img_np.copy()
    for box, cls_id, conf in zip(boxes.xyxy, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # ------------------------------------------------------
    # 2️⃣ EXPLAINABLE AI (GradCAM)
    # ------------------------------------------------------
    heatmap = gradcam_model.run_from_array(img_np)  # Write this helper method below

    # ------------------------------------------------------
    # RETURN EVERYTHING AS JSON
    # ------------------------------------------------------
    return {
        "total_fish": len(class_ids),
        "species_count": species_count,
        "annotated_image": to_base64(annotated_img),
        "heatmap_image": to_base64(heatmap),
        "original_image": to_base64(img_np)
    }
