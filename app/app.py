import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ------------------------------
# LOAD YOLO MODEL
# ------------------------------
MODEL_PATH = "models/fish_yolov8/weights/best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="🐟 Fish Detection App", layout="wide")

st.title("🐟 Underwater Fish Detection System")
st.write("Upload an image and the model will detect fish species with bounding boxes.")

# ------------------------------
# FILE UPLOADER
# ------------------------------
uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ------------------------------
    # RUN MODEL
    # ------------------------------
    results = model.predict(img_array, save=False, conf=0.30)
    result = results[0]

    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    names = result.names

    # ------------------------------
    # COUNT SPECIES
    # ------------------------------
    species_count = {}
    for cid in class_ids:
        label = names.get(cid, "Unknown")
        species_count[label] = species_count.get(label, 0) + 1

    total_fish = len(class_ids)

    # ------------------------------
    # DRAW BOUNDING BOXES
    # ------------------------------
    annotated_img = img_array.copy()

    for box, cls_id, conf in zip(boxes.xyxy, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ------------------------------
    # LAYOUT: 3 COLUMNS IN ONE ROW
    # ------------------------------
    col1, col2, col3 = st.columns([1.3, 1.3, 1])

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Original", width=350)

    with col2:
        st.subheader("🔍 Detection Output")
        st.image(annotated_img, caption="Detected Fish", width=350)

    with col3:
        st.subheader("📊 Summary")
        st.markdown(f"### Total Fish: **{total_fish}**")
        st.write("---")
        st.write("### Species-wise Count")
        for species, count in species_count.items():
            st.markdown(f"- **{species}**: {count}")

    st.success("🎉 Detection Completed Successfully!")
