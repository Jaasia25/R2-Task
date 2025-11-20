# src/train.py
from ultralytics import YOLO
import os
import torch

def train_model():

    # Check GPU status
    print("CUDA Available?:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Name:", torch.cuda.get_device_name(i))
    else:
        print("⚠️ GPU NOT FOUND — training will run on CPU!")

    # Ensure model-saving directory exists
    os.makedirs("models", exist_ok=True)

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Train model on GPU 0
    results = model.train(
        data="data/dataset.yaml",
        epochs=100,
        imgsz=768,
        batch=16,
        patience=15,
        project="models",
        name="fish_yolov8",
        pretrained=True,
        optimizer="Adam",
        lr0=0.0005,
        device="0",       # ← FIXED: use GPU 0
        amp=False         # ← IMPORTANT for GTX 1650 Ti to avoid NaN loss
    )

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    train_model()
