import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from models import TCN_Attention_Model

# =========================
# 1. CONFIG
# =========================
VIDEO_PATH = "../test_video.mov"
YOLO_MODEL_PATH = "yolo11n-10kp.pt"
TCN_MODEL_PATH = "../fall_detection_v4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. MODELLARNI YUKLASH
# =========================
yolo = YOLO(YOLO_MODEL_PATH)
tcn = TCN_Attention_Model(input_size=20).to(DEVICE)
tcn.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
tcn.eval()

def benchmark_pipeline(mode="pose_only"):
    """
    mode: "pose_only" yoki "pose_plus_tcn"
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    pose_buffer = deque(maxlen=30)
    # Buferni to'ldirib qo'yamiz (benchmark barqarorligi uchun)
    for _ in range(30): pose_buffer.append(np.zeros(20, dtype=np.float32))
    
    frame_count = 0
    total_time = 0.0
    
    print(f"🚀 Rejim: {mode} boshlandi...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- O'LCHOV BOSHLANDI ---
            start = time.perf_counter()
            
            # 1. YOLO Pose
            results = yolo.predict(frame, verbose=False, half=True)
            
            # 2. Keypoint Extraction (Ikkala rejimda ham bajariladi!)
            # Chunki real tizimda bu amal baribir kerak
            current_kp = np.zeros(20, dtype=np.float32)
            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                kpts = results[0].keypoints.xyn[0][:10].cpu().numpy().flatten()
                if kpts.shape[0] == 20:
                    current_kp = kpts
            pose_buffer.append(current_kp)

            # 3. TCN (Faqat ikkinchi rejimda)
            if mode == "pose_plus_tcn":
                tcn_in = torch.as_tensor(np.array(pose_buffer), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _ = tcn(tcn_in)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            
            total_time += (time.perf_counter() - start)
            frame_count += 1
            # --- O'LCHOV TUGADI ---

    cap.release()
    fps = frame_count / total_time
    avg_latency = (total_time / frame_count) * 1000
    return fps, avg_latency

# =========================
# 3. TAQQOSLASH (RUN)
# =========================
print("\n=== ILMIY SOLISHTIRUV (UNIFORM PIPELINE) ===")

fps1, lat1 = benchmark_pipeline(mode="pose_only")
print(f"✅ Pose Only     | FPS: {fps1:.2f} | Latency: {lat1:.2f} ms")

fps2, lat2 = benchmark_pipeline(mode="pose_plus_tcn")
print(f"✅ Pose + TCN    | FPS: {fps2:.2f} | Latency: {lat2:.2f} ms")

# =========================
# 4. TAHLIL (PROFESSOR UCHUN)
# =========================
diff_fps = fps1 - fps2
diff_lat = lat2 - lat1

print("\n" + "="*50)
print("📊 TAHLIL NATIJASI:")
print(f"1. TCN qo'shilishi FPSni {diff_fps:.2f} ga kamaytirdi.")
print(f"2. TCN hisobiga kechikish (overhead) bor-yo'g'i {diff_lat:.2f} ms ga oshdi.")
print(f"3. Xulosa: TCN model jami vaqtning { (diff_lat/lat2)*100:.1f}% qismini olmoqda.")
print("="*50)