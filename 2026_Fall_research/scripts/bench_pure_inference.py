import torch
import time
from ultralytics import YOLO
from models import TCN_Attention_Model 
from models_full_kp import TCN_Attention_Model_Full

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODELLAR YO'LI ---
MODEL_PATHS = {
    "10kp_yolo": "yolo11n-10kp.pt",
    "17kp_yolo": "yolo11n-pose.pt",
    "tcn_10kp": "../fall_detection_v4.pth",
    "tcn_17kp": "../fall_detection_full_kp.pth" # Yangi model
}

def measure_pure(yolo_p, tcn_p, kp_num):
    yolo_model = YOLO(yolo_p)
    
    if kp_num == 10:
        tcn_model = TCN_Attention_Model(input_size=20).to(DEVICE)
    else:
        tcn_model = TCN_Attention_Model_Full(input_size=34).to(DEVICE)
    
    tcn_model.load_state_dict(torch.load(tcn_p, map_location=DEVICE))
    tcn_model.eval()

    dummy_img = torch.zeros((1, 3, 640, 640)).to(DEVICE)
    dummy_tcn = torch.zeros((1, 30, kp_num * 2)).to(DEVICE)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = yolo_model(dummy_img, verbose=False)
            _ = tcn_model(dummy_tcn)

    iters = 100
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    
    # Faqat YOLO
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = yolo_model(dummy_img, verbose=False)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    yolo_fps = iters / (time.perf_counter() - t0)

    # YOLO + TCN
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = yolo_model(dummy_img, verbose=False)
            _ = tcn_model(dummy_tcn)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    combined_fps = iters / (time.perf_counter() - t0)

    return yolo_fps, combined_fps

print("\n=== 1. PURE INFERENCE BENCHMARK (PAPER) ===")
f1, f2 = measure_pure(MODEL_PATHS["10kp_yolo"], MODEL_PATHS["tcn_10kp"], 10)
print(f"YOLO 10KP: {f1:.2f} FPS | Combined: {f2:.2f} FPS")

f3, f4 = measure_pure(MODEL_PATHS["17kp_yolo"], MODEL_PATHS["tcn_17kp"], 17)
print(f"YOLO 17KP: {f3:.2f} FPS | Combined: {f4:.2f} FPS")