import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from models_full_kp import TCN_Attention_Model_Full

# =========================
# 1. CONFIG
# =========================
VIDEO_PATH = "../test_video.mov"   # <<< VIDEO FILE
YOLO_MODEL = "yolo11n-pose.pt"
TCN_WEIGHTS = "fall_detection_full_kp.pth"

WINDOW_SIZE = 30
NUM_KP = 17
KP_DIM = 2
TCN_INPUT_SIZE = NUM_KP * KP_DIM  # 34

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Video: {VIDEO_PATH}")
print(f"TCN input: [1, {WINDOW_SIZE}, {TCN_INPUT_SIZE}]")

# =========================
# 2. LOAD MODELS
# =========================
yolo = YOLO(YOLO_MODEL)
tcn = TCN_Attention_Model_Full(input_size=TCN_INPUT_SIZE).to(DEVICE)
tcn.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE))
tcn.eval()

# =========================
# 3. VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Video ochilmadi!"

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# =========================
# 4. BUFFERS
# =========================
pose_buffer = deque(maxlen=WINDOW_SIZE)
prev_kp = np.zeros((NUM_KP, 2), dtype=np.float32)

processed_frames = 0
tcn_calls = 0

# =========================
# 5. WARM-UP
# =========================
dummy = torch.randn(1, WINDOW_SIZE, TCN_INPUT_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(50):
        _ = tcn(dummy)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print("Warm-up done")

# =========================
# 6. BENCHMARK
# =========================
start_time = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- YOLO ----------
        results = yolo(frame, verbose=False)

        kp = prev_kp.copy()

        if (
            results
            and results[0].keypoints is not None
            and results[0].keypoints.xyn is not None
            and results[0].keypoints.xyn.shape[0] > 0
        ):
            kp_all = results[0].keypoints.xyn[0].cpu().numpy()
            kp = kp_all[:NUM_KP]
            prev_kp = kp.copy()

        pose_buffer.append(kp.reshape(-1))
        processed_frames += 1

        # ---------- TCN ----------
        if len(pose_buffer) == WINDOW_SIZE:
            tcn_input = torch.tensor(
                np.array(pose_buffer),
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            _ = tcn(tcn_input)
            tcn_calls += 1

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

# =========================
# 7. RESULTS
# =========================
total_time = end_time - start_time
fps = processed_frames / total_time
avg_latency_ms = (total_time / processed_frames) * 1000

print("\n" + "="*40)
print("VIDEO + YOLO (17kp) + TCN BENCHMARK")
print(f"Processed frames : {processed_frames}")
print(f"TCN calls        : {tcn_calls}")
print(f"Total time       : {total_time:.2f} sec")
print(f"End-to-End FPS   : {fps:.2f}")
print(f"Avg latency/frame: {avg_latency_ms:.2f} ms")
print("="*40)

cap.release()
