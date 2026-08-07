import cv2
import torch
import numpy as np
from ultralytics import YOLO
import collections
import time

# Model klasslarini import qilish
from models import TCN_Attention_Model        
from models_full_kp import TCN_Attention_Model_Full

# =========================
# 1. UNIVERSAL SOZLAMALAR
# =========================
MODE = "pretrained"  # "custom" yoki "pretrained"
VIDEO_PATH = "../test_video.mov"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30

if MODE == "custom":
    YOLO_MODEL_PATH = "yolo11n-10kp.pt"
    TCN_MODEL_PATH = "../fall_detection_v4.pth"
    NUM_KP = 10
    TCN_CLASS = TCN_Attention_Model
else:
    YOLO_MODEL_PATH = "yolo11n-pose.pt"
    TCN_MODEL_PATH = "fall_detection_full_kp.pth"
    NUM_KP = 17
    TCN_CLASS = TCN_Attention_Model_Full

TCN_INPUT_SIZE = NUM_KP * 2 

# =========================
# 2. MODELLARNI YUKLASH VA WARM-UP
# =========================
print(f"📦 Modellarni yuklash ({DEVICE})...")
yolo = YOLO(YOLO_MODEL_PATH)
tcn = TCN_CLASS(input_size=TCN_INPUT_SIZE).to(DEVICE)
tcn.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
tcn.eval()

# --- WARM-UP QISMI ---
print(f"🔥 Warm-up boshlandi (bu bir necha soniya olishi mumkin)...")
with torch.no_grad():
    # 1. YOLO Warm-up (bo'sh rasm bilan)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(20):
        _ = yolo(dummy_img, verbose=False)
    
    # 2. TCN Warm-up (bo'sh tensor bilan)
    dummy_tensor = torch.randn(1, WINDOW_SIZE, TCN_INPUT_SIZE).to(DEVICE)
    for _ in range(50):
        _ = tcn(dummy_tensor)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
print("✅ Warm-up yakunlandi. Video boshlanmoqda...")

# =========================
# 3. VIDEO VA BUFERLAR
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
pose_buffer = collections.deque(maxlen=WINDOW_SIZE)
prev_kp = np.zeros(TCN_INPUT_SIZE, dtype=np.float32)

frame_count = 0
bench_start_time = None
prev_time = time.perf_counter()
current_status = "System Ready"
status_color = (0, 255, 0)

# =========================
# 4. ASOSIY SIKL
# =========================
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        current_input = prev_kp.copy()

        # --- YOLO INFERENCE ---
        results = yolo(frame, verbose=False)

        if results and results[0].keypoints is not None:
            xyn = results[0].keypoints.xyn
            xy_pixel = results[0].keypoints.xy

            if xyn is not None and xyn.shape[0] > 0:
                kp_norm = xyn[0].cpu().numpy()
                
                if kp_norm.shape[0] >= NUM_KP:
                    current_input = kp_norm[:NUM_KP].flatten()
                    prev_kp = current_input.copy()

                    # --- FAQAT YASHIL NUQTALAR ---
                    kp_pixel = xy_pixel[0].cpu().numpy()
                    for pt in kp_pixel[:NUM_KP]:
                        cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

        pose_buffer.append(current_input)

        # --- TCN INFERENCE ---
        if len(pose_buffer) == WINDOW_SIZE:
            tcn_in = torch.as_tensor(np.array([pose_buffer]), dtype=torch.float32).to(DEVICE)
            output = tcn(tcn_in)
            
            prob = torch.softmax(output, dim=1)
            conf, class_idx = torch.max(prob, dim=1)
            
            if class_idx.item() == 1 and conf.item() > 0.7:
                current_status = f"FALL! ({conf.item():.2f})"
                status_color = (0, 0, 255)
            else:
                current_status = "Normal"
                status_color = (0, 255, 0)

        # --- FPS BENCHMARK (Warm-up'dan keyin boshlanadi) ---
        if frame_count == 1: # Video boshlanishi bilan vaqtni o'lchaymiz
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            bench_start_time = time.perf_counter()

        curr_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Dashboard
        cv2.rectangle(frame, (5, 5), (380, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"MODE: {MODE.upper()}", (15, 35), 2, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"STATUS: {current_status}", (15, 70), 2, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (280, 35), 2, 0.6, (0, 255, 255), 1)

        cv2.imshow("Optimized Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

# =========================
# 5. YAKUNIY NATIJA
# =========================
if bench_start_time:
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    total_duration = time.perf_counter() - bench_start_time
    avg_fps = frame_count / total_duration
    print(f"\n📊 FINAL BENCHMARK ({MODE.upper()})")
    print(f"✅ AVG FPS: {avg_fps:.2f}")
    print(f"Latency: {(total_duration/frame_count)*1000:.2f} ms")

cap.release()
cv2.destroyAllWindows()