import time
import torch
from ultralytics import YOLO

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Model
model = YOLO("yolo11n-10kp.pt")
model.to(device)

# Dummy input
dummy = torch.randn(1, 3, 640, 640).to(device)

# Warmup
for _ in range(10):
    model(dummy)

# FPS test
N = 100
t0 = time.time()
for _ in range(N):
    model(dummy)
t1 = time.time()

fps = N / (t1 - t0)
print(f"YOLO Pose FPS: {fps:.2f}")
