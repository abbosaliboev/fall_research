import os
from ultralytics import YOLO

# YOLOv8-n configuration
MODEL_NAME = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

DATA_YAML = r"C:\Users\ali\Projects\fall_research\ppe\data\data.yaml"
PROJECT_DIR = "runs_ppe"
RUN_NAME = "yolov8n_scratch"

def ensure_model():
    if not os.path.exists(MODEL_NAME):
        print(f"[INFO] '{MODEL_NAME}' not found. Downloading...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print("[INFO] Download complete:", MODEL_NAME)
    else:
        print("[INFO] Model already exists:", MODEL_NAME)

def main():
    ensure_model()

    print("[INFO] Loading YOLOv8-n model...")
    model = YOLO(MODEL_NAME)

    print("[INFO] Starting training...")
    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=100,
        batch=16,
        workers=0,
        device=0,
        pretrained=False,   # YOLOv8 uchun ham scratch variant
        project=PROJECT_DIR,
        name=RUN_NAME
    )

if __name__ == "__main__":
    main()
