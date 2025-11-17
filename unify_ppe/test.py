from ultralytics import YOLO

m = YOLO("yolo11s.pt")
for i, layer in enumerate(m.model.model):
    print(i, type(layer).__name__)
