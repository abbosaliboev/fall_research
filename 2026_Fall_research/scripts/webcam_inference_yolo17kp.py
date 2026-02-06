import cv2
import torch
import numpy as np
from ultralytics import YOLO
from models_full_kp import TCN_Attention_Model_Full 
import collections
import time
from threading import Thread

# --- SOZLAMALAR ---
MAX_BENCH_FRAMES = 1000 # Warm-up'dan keyin aynan 200 kadr o'lchanadi
WARMUP_FRAMES = 200    # Isib olish uchun 100 kadr yetarli

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- MODELLAR ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('yolo11n-pose.pt')
tcn_model = TCN_Attention_Model_Full(input_size=34).to(DEVICE)
tcn_model.load_state_dict(torch.load('fall_detection_full_kp.pth', map_location=DEVICE))
tcn_model.eval()

frame_buffer = collections.deque(maxlen=30)
frame_count = 0
bench_frame_count = 0
bench_start_time = None
prev_time = 0

vs = VideoStream(src=0).start()
time.sleep(2.0)

print(f"🚀 17-KP Benchmark boshlandi. (Limit: {MAX_BENCH_FRAMES} kadr)")

while True:
    frame = vs.read()
    if frame is None: break

    frame_count += 1
    
    # Inference
    results = yolo_model(frame, verbose=False)
    if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
        kp_norm = results[0].keypoints.xyn[0].cpu().numpy()
        if len(kp_norm) == 17:
            frame_buffer.append(kp_norm.flatten())
            if len(frame_buffer) == 30:
                input_array = np.expand_dims(np.array(frame_buffer), axis=0)
                input_tensor = torch.from_numpy(input_array).float().to(DEVICE)
                with torch.no_grad():
                    _ = tcn_model(input_tensor)

    # --- BENCHMARK LOGIKASI ---
    if frame_count > WARMUP_FRAMES:
        if bench_start_time is None:
            bench_start_time = time.time()
            print("🔥 Warm-up tugadi. O'lchash boshlandi...")

        bench_frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"17-KP Bench: {bench_frame_count}/{MAX_BENCH_FRAMES}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Warm-up: {frame_count}/{WARMUP_FRAMES}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("17-KP Benchmark", frame)

    if bench_frame_count >= MAX_BENCH_FRAMES or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Yakuniy natija
if bench_start_time:
    total_time = time.time() - bench_start_time
    print("\n" + "="*40)
    print(f"📊 17-KP NATIJALARI")
    print(f"Jami kadrlar: {bench_frame_count}")
    print(f"Vaqt: {total_time:.2f} sek")
    print(f"✅ AVG FPS: {bench_frame_count / total_time:.2f}")
    print("="*40)

vs.stop()
cv2.destroyAllWindows()