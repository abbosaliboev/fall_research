import cv2
import torch
import numpy as np
from ultralytics import YOLO
from models import TCN_Attention_Model
import collections
import time
from threading import Thread

# --- SOZLAMALAR ---
MAX_BENCH_FRAMES = 1000 # Warm-up'dan keyin aynan 200 kadr o'lchanadi
WARMUP_FRAMES = 200    # Isib olish uchun 100 kadr yetarli (1000 juda ko'plik qiladi)

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
YOLO_PATH = 'yolo11n-10kp.pt' 
TCN_PATH = '../fall_detection_v4.pth'

yolo_model = YOLO(YOLO_PATH)
tcn_model = TCN_Attention_Model(input_size=20).to(DEVICE)
tcn_model.load_state_dict(torch.load(TCN_PATH, map_location=DEVICE))
tcn_model.eval()

frame_buffer = collections.deque(maxlen=30)
frame_count = 0
bench_frame_count = 0
bench_start_time = None
prev_time = 0

vs = VideoStream(src=0).start()
time.sleep(2.0)

print(f"🚀 10-KP Benchmark boshlandi. (Limit: {MAX_BENCH_FRAMES} kadr)")

while True:
    frame = vs.read()
    if frame is None: break

    frame_count += 1
    
    # --- YOLO INFERENCE ---
    results = yolo_model(frame, verbose=False)
    
    if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
        kp_norm = results[0].keypoints.xyn[0].cpu().numpy()
        
        if len(kp_norm) == 10:
            final_input = kp_norm.flatten()
            frame_buffer.append(final_input)
            
            if len(frame_buffer) == 30:
                input_array = np.expand_dims(np.array(frame_buffer), axis=0)
                input_tensor = torch.from_numpy(input_array).float().to(DEVICE)
                
                with torch.no_grad():
                    output = tcn_model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf, class_idx = torch.max(prob, dim=1)
                    
                    label = "FALL!" if class_idx.item() == 1 and conf.item() > 0.8 else "Normal"
                    color = (0, 0, 255) if label == "FALL!" else (0, 255, 0)
                    cv2.putText(frame, f"STATUS: {label}", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # --- BENCHMARK LOGIKASI ---
    if frame_count > WARMUP_FRAMES:
        if bench_start_time is None:
            bench_start_time = time.time()
            print("🔥 Warm-up tugadi. 10-KP o'lchash boshlandi...")

        bench_frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"10-KP Bench: {bench_frame_count}/{MAX_BENCH_FRAMES}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Warming up: {frame_count}/{WARMUP_FRAMES}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("10-KP Benchmark", frame)

    # Avtomatik to'xtash sharti
    if bench_frame_count >= MAX_BENCH_FRAMES or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- YAKUNIY STATISTIKA ---
if bench_start_time is not None:
    total_bench_time = time.time() - bench_start_time
    avg_fps = bench_frame_count / total_bench_time
    print("\n" + "="*40)
    print(f"📊 10-KP FINAL RESULTS")
    print(f"Ishlangan kadrlar: {bench_frame_count}")
    print(f"Benchmark vaqti: {total_bench_time:.2f} sek")
    print(f"✅ O'RTACHA BARQAROR FPS: {avg_fps:.2f}")
    print("="*40)

vs.stop()
cv2.destroyAllWindows()