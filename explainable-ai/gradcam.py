# src/gradcam.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_gradcam(model_path, image_path, target_class=None):
    """
    Generates Grad-CAM heatmap for YOLOv8 detection model.
    
    Args:
        model_path: Path to trained YOLOv8 weights
        image_path: Input image path
        target_class: Optional, only show Grad-CAM for this class
    """
    # -------------------------------
    # Load model
    # -------------------------------
    model = YOLO(model_path)
    model.model.eval()

    # -------------------------------
    # Load image
    # -------------------------------
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    input_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor.requires_grad = True

    # -------------------------------
    # Forward pass
    # -------------------------------
    with torch.no_grad():
        results = model.predict(img_array, conf=0.3)

    result = results[0]
    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    names = result.names

    # -------------------------------
    # Grad-CAM
    # -------------------------------
    # Pick the last conv layer of YOLOv8 backbone
    target_layer = model.model.model[17].m[-1]  # adjust depending on your model version

    # Forward pass with hooks
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # For each detected object
    for i, cls_id in enumerate(class_ids):
        if target_class and names[cls_id] != target_class:
            continue

        # Compute loss as sum of predicted object score
        score = boxes.conf[i]
        score.backward(retain_graph=True)

        # Compute Grad-CAM
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        cam = torch.zeros(features.shape[2:], dtype=torch.float32)

        for j in range(features.shape[1]):
            cam += pooled_gradients[j] * features[0, j]

        cam = np.maximum(cam.detach().cpu().numpy(), 0)
        cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Overlay heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"{names[cls_id]} {boxes.conf[i]:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"Grad-CAM: {names[cls_id]}")
        plt.show()

    handle_fw.remove()
    handle_bw.remove()

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    MODEL_PATH = "models/fish_yolov8/weights/best.pt"
    IMAGE_PATH = "data/sample_fish.jpg"
    show_gradcam(MODEL_PATH, IMAGE_PATH)
