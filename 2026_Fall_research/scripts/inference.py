import cv2
import torch
import numpy as np
from ultralytics import YOLO
from models import TCN_Attention_Model
import collections
import time

# 1. Qurilma va Modellarni yuklash
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('yolo11n-pose.pt')

model = TCN_Attention_Model().to(DEVICE)
model.load_state_dict(torch.load("../fall_detection_v4.pth", map_location=DEVICE))
model.eval()

# 2. Parametrlar va Vizualizatsiya sozlamalari
WANTED_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]
SKELETON_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12), # Tana (Torso)
    (11, 13), (13, 15), # Chap oyoq
    (12, 14), (14, 16)  # O'ng oyoq
]

frame_buffer = collections.deque(maxlen=30)
fall_counter = 0 
prev_frame_time = 0

video_path = "../test_video.mov"
cap = cv2.VideoCapture(video_path)

print(f"Inference boshlandi... Qurilma: {DEVICE}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    annotated_frame = frame.copy()
    yolo_time, tcn_time = 0, 0
    conf = 0.0

    # --- 1. YOLO INFERENCE (Pose Estimation) ---
    t_start_yolo = time.time()
    results = yolo_model(frame, verbose=False)
    yolo_time = (time.time() - t_start_yolo) * 1000 # ms

    if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
        kp_norm = results[0].keypoints.xyn[0].cpu().numpy()
        kp_pixel = results[0].keypoints.xy[0].cpu().numpy()
        
        if len(kp_norm) >= 17:
            # --- 2. SKELETNI CHIZISH (Faqat 10 ta nuqta) ---
            # Neck (Bo'yin) hisoblash va chizish
            neck_px = (kp_pixel[5] + kp_pixel[6]) / 2
            cv2.circle(annotated_frame, (int(neck_px[0]), int(neck_px[1])), 6, (0, 255, 255), -1)

            # Tanlangan 10 ta nuqtani chizish
            for idx in WANTED_KP:
                x, y = int(kp_pixel[idx][0]), int(kp_pixel[idx][1])
                cv2.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)

            # Suyaklarni chizish
            for start, end in SKELETON_CONNECTIONS:
                pt1 = (int(kp_pixel[start][0]), int(kp_pixel[start][1]))
                pt2 = (int(kp_pixel[end][0]), int(kp_pixel[end][1]))
                if pt1[0] > 0 and pt2[0] > 0:
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)

            # --- 3. TCN INFERENCE (Fall Detection) ---
            filtered_kp = kp_norm[WANTED_KP]
            neck_x = (kp_norm[5][0] + kp_norm[6][0]) / 2
            neck_y = (kp_norm[5][1] + kp_norm[6][1]) / 2
            final_row = list(filtered_kp.flatten()) + [neck_x, neck_y]
            frame_buffer.append(final_row)
            
            if len(frame_buffer) == 30:
                input_data = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(DEVICE)
                
                t_start_tcn = time.time()
                with torch.no_grad():
                    output = model(input_data)
                    prediction = torch.softmax(output, dim=1)
                    prob, class_idx = torch.max(prediction, dim=1)
                tcn_time = (time.time() - t_start_tcn) * 1000 # ms
                
                conf = prob.item()
                if class_idx.item() == 1 and conf > 0.85:
                    fall_counter += 1
                else:
                    fall_counter = max(0, fall_counter - 1)

    # Status va FPS hisoblash
    status = "FALLING!" if fall_counter >= 5 else "Normal"
    color = (0, 0, 255) if status == "FALLING!" else (0, 255, 0)
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # --- 4. MA'LUMOTLARNI EKRANGA CHIQARISH ---
    # Asosiy Status
    cv2.putText(annotated_frame, f"{status} ({conf:.2f})", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Tezlik ko'rsatkichlari (ms)
    cv2.rectangle(annotated_frame, (20, 75), (320, 200), (0,0,0), -1) # Fon
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"YOLO speed: {yolo_time:.1f} ms", (30, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(annotated_frame, f"TCN speed: {tcn_time:.1f} ms", (30, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(annotated_frame, f"Fall Counter: {fall_counter}", (30, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Fall Detection - Research & Efficiency Test", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()