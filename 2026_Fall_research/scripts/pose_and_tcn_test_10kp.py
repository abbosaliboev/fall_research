import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from models import TCN_Attention_Model

# =========================
# 1. SOZLAMALAR (CONFIG)
# =========================
VIDEO_PATH = "../test_video.mov"
YOLO_MODEL_PATH = "yolo11n-10kp.pt"
TCN_MODEL_PATH = "../fall_detection_v4.pth"

WINDOW_SIZE = 30
NUM_KP = 10
TCN_INPUT_SIZE = NUM_KP * 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. MODELLARNI YUKLASH
# =========================
yolo = YOLO(YOLO_MODEL_PATH)
tcn = TCN_Attention_Model(input_size=TCN_INPUT_SIZE).to(DEVICE)
tcn.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
tcn.eval()

# Warm-up (GPU faollashtirish)
dummy_tcn = torch.randn(1, WINDOW_SIZE, TCN_INPUT_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(20):
        _ = tcn(dummy_tcn)
        _ = yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

# =========================
# 3. VIDEO VA BUFERLAR
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
pose_buffer = deque(maxlen=WINDOW_SIZE)
prev_kp = np.zeros(TCN_INPUT_SIZE, dtype=np.float32)

processed_frames = 0
tcn_calls = 0
stride = 2  # Har 2-kadrda TCN hisoblanadi

print(f"🚀 Turbo Benchmark boshlandi (Device: {DEVICE}, Stride: {stride})")

# =========================
# 4. ASOSIY SIKL
# =========================
start_time = time.perf_counter()

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        
        # --- YOLO INFERENCE ---
        # half=True GPU-da tezlikni sezilarli oshiradi
        results = yolo.predict(frame, verbose=False, half=True)
        
        # --- OPTIMALLASHGAN DATA FLOW ---
        # Har doim ham ma'lumotni CPUga tortmaymiz (PCIe yukini kamaytirish)
        if processed_frames % stride == 0:
            current_kp = prev_kp
            
            if results and results[0].keypoints is not None:
                kpts_tensor = results[0].keypoints.xyn
                if kpts_tensor is not None and kpts_tensor.shape[0] > 0:
                    # CPUga o'tkazish faqat shu yerda
                    current_kp = kpts_tensor[0][:NUM_KP].cpu().numpy().flatten()
                    if current_kp.shape[0] == TCN_INPUT_SIZE:
                        prev_kp = current_kp
            
            pose_buffer.append(current_kp)

            # --- TCN INFERENCE ---
            if len(pose_buffer) == WINDOW_SIZE:
                # as_tensor + pre-allocated device memory
                tcn_in = torch.as_tensor(np.array(pose_buffer), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _ = tcn(tcn_in)
                tcn_calls += 1

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

# =========================
# 5. NATIJALAR
# =========================
total_time = end_time - start_time
fps = processed_frames / total_time

print("\n" + "="*40)
print("🏁 FINAL TURBO BENCHMARK RESULTS")
print(f"Jami kadrlar      : {processed_frames}")
print(f"TCN chaqiruvlari  : {tcn_calls}")
print(f"Umumiy vaqt       : {total_time:.2f} sek")
print(f"✅ END-TO-END FPS : {fps:.2f}")
print(f"Latency per frame : {(total_time/processed_frames)*1000:.2f} ms")
print("="*40)

cap.release()