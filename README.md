# 🌊 Underwater Fish Detection and Monitoring System

This project implements a complete, end-to-end computer vision pipeline for automated underwater fish detection and species monitoring.

The system utilizes the state-of-the-art YOLOv8 architecture for robust object detection, incorporates Grad-CAM for Explainable AI (XAI) insights, and is deployed via a modern, scalable web stack consisting of a Streamlit frontend and a dedicated FastAPI backend.

## ✨ Features

- **Automated Species Detection**: Detects and classifies seven different underwater species (including various fish, sharks, penguins, puffins, jellyfish, starfish, and stingrays) using a fine-tuned YOLOv8 model.

- **Explainable AI (XAI)**: Integrated Grad-CAM visualization shows the spatial regions in the image that most influenced the model's detection confidence.

- **Scalable Deployment**: Decoupled web application architecture featuring a lightweight Streamlit interface for user interaction and a high-performance, asynchronous FastAPI backend for model inference.

- **Robust Data Pipeline**: Comprehensive data preprocessing and aggressive augmentation strategies (including simulated underwater effects like dehazing and fog) to improve model generalization.

- **Iterative Optimization**: Seven preliminary models were tested to identify the optimal architecture and hyperparameter configuration (YOLOv8s @ 768px input) for production efficiency and performance.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

Ensure you have Python 3.9+ and Git installed.

### 2. Clone the Repository

```bash
git clone <repository_url>
cd underwater-fish-detection
```

### 3. Set up the Environment

The project is designed to run in a dedicated virtual environment.

```bash
# 1. Create a virtual environment named 'fish'
python -m venv fish

# 2. Activate the environment
# On macOS/Linux:
source fish/bin/activate
# On Windows:
.\fish\Scripts\activate

# 3. Upgrade pip to the latest version
pip install --upgrade pip

# 4. Install dependencies from requirements.txt
pip install -r requirements.txt

```

### 4. Download Model Weights

The required final fine-tuned model weights (`yolov8s.pt` or `yolov8m.pt`) should be placed in the project root directory.

**Weights Location**: Check the `models/` directory for saved checkpoints from the iterative training process. If the final model weights are not in the root directory, they can be retrieved from the `models/` folder (e.g., `models/fish_yolov87/weights/best.pt`).

## 💻 Usage

The application requires two components to run: the FastAPI inference server and the Streamlit frontend.

### 1. Run the FastAPI Backend (Inference Server)

This server hosts the YOLOv8 model and the Grad-CAM logic, handling all heavy computational tasks.

```bash
# Run the FastAPI server with auto-reload for development
# It will typically run on http://127.0.0.1:8000
uvicorn app.api:app --reload
```

### 2. Run the Streamlit Frontend

With the FastAPI server running, launch the Streamlit application in a separate terminal window.

```bash
# Run the Streamlit UI
# It will typically open in your browser on http://localhost:8501
streamlit run streamlit-app/app.py
```

### Application Workflow

1. Navigate to the Streamlit URL (http://localhost:8501)
2. Upload an image file (PNG or JPG) containing marine life
3. The Streamlit app sends the image to the FastAPI backend
4. The backend runs YOLOv8 detection and Grad-CAM generation
5. The backend returns the processed image, heatmap, and structured JSON results
6. The Streamlit app displays the bounding box image, the XAI heatmap, and a summary of species counts and confidence scores

## 📂 Project Structure

The project follows a standard machine learning repository layout, optimizing for modularity and deployment.

| Directory / File | Description |
|------------------|-------------|
| `requirements.txt` | Lists all necessary Python packages and dependencies |
| `yolov8s.pt`, `yolov8m.pt` | Model weight files used for inference |
| `config/` | Stores YAML configuration for training and model parameters |
| └── `config.yaml` | Defines dataset paths, class names, and hyperparameters |
| `data/` | Stores the dataset, including images, labels, and training splits |
| `models/` | Contains various checkpoints from the iterative training process |
| `src/` | Core scripts for model development |
| └── `train.py` | Primary script for model training and experimentation |
| `notebook/` | Jupyter Notebooks for analysis and prototyping |
| ├── `data_preprocessing.ipynb` | Detailed steps for data preparation and augmentation |
| ├── `gradcam.ipynb` | Prototyping script for the XAI visualization logic |
| └── `test_data.ipynb` | Notebook for quick inference checks and test set evaluation |
| `app/` | Contains the high-performance model serving backend (FastAPI) |
| └── `api.py` | FastAPI endpoint for detection, heatmap generation, and results structuring |
| `streamlit-app/` | Contains the user-facing frontend (Streamlit) |
| ├── `app.py` | Streamlit main application script for upload and display |
| └── `uploads/` | Directory for temporary storage of user-uploaded images |
| `utils/` | General purpose helper functions |
| └── `utils.py` | Functions for image handling, result parsing, and common tasks |
| `xai/` | Specialized scripts for model explainability |
| └── `gradcam.py` | Implementation of the Grad-CAM algorithm tailored for YOLOv8 |

## 🛠 Model and Methodology

### Dataset

- **Base Dataset**: Aquarium and Marine Life (Roboflow)
- **Expansion**: Custom-annotated images (specifically for shark, penguin, and puffin) were merged to create a final training set of 958 images
- **Classes**: 7 distinct species categories

### Training Details

- **Architecture**: YOLOv8-small (`yolov8s`)
- **Input Resolution**: 768×768 pixels (determined as optimal after iterative testing)
- **Key Augmentations**: Underwater Dehaze, Random Fog, ColorJitter, and CLAHE to simulate real-world underwater environments

### Performance (Test Set Evaluation)

| Metric | Value | Description |
|--------|-------|-------------|
| mAP@0.5 | 0.759 | Mean Average Precision at IoU threshold of 0.5 |
| mAP@0.5:0.95 | 0.469 | Generalization score averaged over IoU thresholds from 0.5 to 0.95 |