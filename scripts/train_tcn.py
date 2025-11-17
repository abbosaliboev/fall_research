import torch
import torch.nn as nn
import cv2
import numpy as np
from multiprocessing import Process, Queue
import time
import os

# ========== MODEL DEFINITIONS (same as training) ==========
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        pad = ((k - 1) * dil) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in,  c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, in_ch, num_classes=3, drop=0.2):
        super().__init__()
        dilations = [1, 2, 4, 8]
        channels  = [64, 128, 256, 256]
        layers = []
        c = in_ch
        for out_c, d in zip(channels, dilations):
            layers.append(TCNBlock(c, out_c, k=3, dil=d, drop=drop))
            c = out_c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, num_classes)
        )

    def forward(self, x):  # x: [B, C, L]
        y = self.tcn(x)
        y = self.head(y)
        return y

# ========== CAMERA READER ==========
def camera_reader(rtsp_url, q, cam_id):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[Cam {cam_id}] ❌ Cannot open RTSP stream.")
        return
    print(f"[Cam {cam_id}] ✅ Started stream.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[Cam {cam_id}] ⚠️ Frame read failed.")
            break
        # Resize and normalize (for demo we use dummy features)
        frame_small = cv2.resize(frame, (64, 64))
        features = np.mean(frame_small, axis=(0, 1))  # fake feature vector (3,)
        q.put((cam_id, features))
        time.sleep(0.05)
    cap.release()
    print(f"[Cam {cam_id}] ⏹️ Stopped.")

# ========== PREDICTOR ==========
def predictor(q, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Predictor] Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    in_ch = ckpt.get("in_ch", 34)
    state_dict = ckpt.get("model", ckpt)

    model = TCN(in_ch=in_ch, num_classes=3)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("[Predictor] ✅ Model loaded successfully.")

    # Buffer for each camera (fake input of shape [C, L])
    buffers = {i: [] for i in range(4)}
    SEQ_LEN = 30  # same as training seq_len
    LABELS = ["No Fall", "Pre Fall", "Fall"]

    while True:
        try:
            cam_id, feat = q.get(timeout=5)
        except:
            continue
        buffers[cam_id].append(feat)
        if len(buffers[cam_id]) >= SEQ_LEN:
            x = np.array(buffers[cam_id][-SEQ_LEN:]).T  # [C, L]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x_tensor)
                pred = torch.argmax(logits, dim=1).item()
            print(f"[Cam {cam_id}] → {LABELS[pred]}")
            # Only keep last SEQ_LEN frames
            buffers[cam_id] = buffers[cam_id][-SEQ_LEN:]

# ========== MAIN ==========
if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\ali\Projects\fall_research\experiments\FD-01\best-5-80.pt"

    # RTSP links
    RTSP_LINKS = [
        {"id": "cam1", "url": "rtsp://admin:Aminok434*@10.198.137.226:554/stream1"},
        {"id": "cam2", "url": "rtsp://admin:Aminok434*@10.198.137.226:554/stream1"},
        {"id": "cam3", "url": "rtsp://admin:Aminok434*@10.198.137.226:554/stream1"},
        {"id": "cam4", "url": "rtsp://admin:Aminok434*@10.198.137.226:554/stream1"},
    ]

    q = Queue()
    procs = []

    # Start camera readers
    for i, rtsp in enumerate(RTSP_LINKS):
        p = Process(target=camera_reader, args=(rtsp, q, i))
        p.start()
        procs.append(p)

    # Start predictor
    pred_proc = Process(target=predictor, args=(q, MODEL_PATH))
    pred_proc.start()
    procs.append(pred_proc)

    print("[Main] ✅ Started all RTSP readers and predictor.")

    for p in procs:
        p.join()
