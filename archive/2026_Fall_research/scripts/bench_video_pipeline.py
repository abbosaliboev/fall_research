import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from models import TCN_Attention_Model
from models_full_kp import TCN_Attention_Model_Full

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_PATH = "../test_video.mov"

def run_video_test(yolo_p, tcn_p, kp_num, with_tcn=False):
    yolo = YOLO(yolo_p)
    tcn_model = None
    
    if with_tcn:
        if kp_num == 10:
            tcn_model = TCN_Attention_Model(input_size=20).to(DEVICE)
        else:
            tcn_model = TCN_Attention_Model_Full(input_size=34).to(DEVICE)
        tcn_model.load_state_dict(torch.load(tcn_p, map_location=DEVICE))
        tcn_model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # YOLO
            results = yolo.predict(frame, verbose=False, half=True)
            
            # TCN (Sinxron ishlov berish)
            if with_tcn:
                if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                    kp = results[0].keypoints.xyn[0][:kp_num].cpu().numpy().flatten()
                    if len(kp) == kp_num * 2:
                        tcn_in = torch.zeros((1, 30, kp_num * 2), device=DEVICE)
                        _ = tcn_model(tcn_in)
            
            frames += 1
            
    total_time = time.perf_counter() - start_time
    cap.release()
    return frames / total_time

print("\n=== 2. VIDEO PIPELINE BENCHMARK (DEMO) ===")
# 10KP
v1 = run_video_test("yolo11n-10kp.pt", "../fall_detection_v4.pth", 10, False)
v2 = run_video_test("yolo11n-10kp.pt", "../fall_detection_v4.pth", 10, True)
print(f"10KP Video - Pose: {v1:.2f} | Combined: {v2:.2f}")

# 17KP
v3 = run_video_test("yolo11n-pose.pt", "../fall_detection_full_kp.pth", 17, False)
v4 = run_video_test("yolo11n-pose.pt", "../fall_detection_full_kp.pth", 17, True)
print(f"17KP Video - Pose: {v3:.2f} | Combined: {v4:.2f}")