# scripts/test_2.py
import os
import cv2
import torch
import numpy as np
import time
from threading import Thread
import queue
from ultralytics import YOLO
from tcn_model import TCN  # TCN arxitekturang shu faylda
# Use same normalization as feature extraction pipeline
from extract_pose import _normalize

# ====== PATHLAR ======
PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
TCN_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "FD-01", "best-1-60.pt")
YOLO_MODEL_PATH = "yolov8n-pose.pt"

# ==== Kamera manzillarini shu yerga yoz ====
CAMERA_SOURCES = [
    {"id": "cam1", "url": "rtsp://admin:Aminok434*@10.198.137.222:554/stream1"},
    {"id": "cam2", "url": "rtsp://admin:Aminok434*@10.198.137.221:554/stream1"},
]

SEQUENCE_LENGTH = 15
LABEL_MAP = {0: 'no_fall', 1: 'pre_fall', 2: 'fall'}
IMG_SIZE = 448
CONF_THRESH = 0.3

# YOLO Pose keypoint connections (COCO format)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Bosh
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Qo'llar
    (5, 11), (6, 12), (11, 12),  # Tana
    (11, 13), (13, 15), (12, 14), (14, 16)  # Oyoqlar
]


# ==== CAMERA READER ====
class CameraReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Kamera ochilmadi: {src}")
        self.q = queue.Queue(maxsize=1)
        self.running = True
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            if not self.q.empty():
                self.q.get()
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()


def draw_pose(frame, keypoints, label):
    """
    Odamning skeleton va bounding box chizish
    """
    # Rangni aniqlash
    if label == "fall":
        color = (0, 0, 255)  # Qizil
        bbox_color = (0, 0, 255)
    elif label == "pre_fall":
        color = (0, 165, 255)  # To'q sariq
        bbox_color = (0, 165, 255)
    else:
        color = (0, 255, 0)  # Yashil
        bbox_color = (0, 255, 0)
    
    # Keypoint'larni chizish
    valid_points = []
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # Valid keypoint
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            valid_points.append((x, y))
    
    # Skeleton lines chizish
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            # Agar har ikki point ham valid bo'lsa
            if start_point[0] > 0 and start_point[1] > 0 and \
               end_point[0] > 0 and end_point[1] > 0:
                cv2.line(frame, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        color, 2)
    
    # Bounding box chizish
    if len(valid_points) > 0:
        valid_points = np.array(valid_points)
        x_min, y_min = valid_points.min(axis=0)
        x_max, y_max = valid_points.max(axis=0)
        
        # Margin qo'shish
        margin = 20
        x_min = max(0, int(x_min - margin))
        y_min = max(0, int(y_min - margin))
        x_max = min(frame.shape[1], int(x_max + margin))
        y_max = min(frame.shape[0], int(y_max + margin))
        
        # Box chizish
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)
        
        # Label yozish
        label_text = f"{label.upper()}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (x_min, y_min - text_height - baseline - 10),
                     (x_min + text_width + 10, y_min),
                     bbox_color, -1)
        
        # Text
        cv2.putText(frame, label_text,
                   (x_min + 5, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ==== MAIN FUNCTION ====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] ✅ Using device: {device}")

    # Load YOLO pose model
    yolo = YOLO(YOLO_MODEL_PATH)

    # ====== Load TCN model ======
    print("[Predictor] 🔄 Loading TCN model...")
    ckpt = torch.load(TCN_MODEL_PATH, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    in_ch = ckpt.get("in_ch", 34)
    tcn = TCN(input_size=in_ch, output_size=3, num_channels=[64, 128, 256])
    missing, unexpected = tcn.load_state_dict(state_dict, strict=False)
    print(f"[Predictor] ✅ Model loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    tcn.to(device).eval()

    # ==== Start cameras ====
    cameras = []
    for cam in CAMERA_SOURCES:
        try:
            reader = CameraReader(cam["url"])
            cameras.append({"id": cam["id"], "reader": reader, "seq": []})
            print(f"[Main] ✅ Started camera: {cam['id']}")
        except Exception as e:
            print(f"[Main] ❌ {cam['id']} kamerani ochib bo'lmadi: {e}")

    print(f"[Main] ✅ Total {len(cameras)} camera readers started.")

    # ==== Process each camera ====
    try:
        while True:
            for cam in cameras:
                frame = cam["reader"].read()
                if frame is None:
                    continue

                # Nusxa ko'chirish (original frame buzilmasligi uchun)
                display_frame = frame.copy()

                results = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
                if len(results) == 0 or len(results[0].keypoints) == 0:
                    # Agar odam topilmasa
                    cv2.putText(display_frame, "No person detected", (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.imshow(cam["id"], display_frame)
                    continue

                # Birinchi odamning keypoint'lari (raw) va confidences
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]     # (17,2)
                kp_conf   = results[0].keypoints.conf.cpu().numpy()[0]   # (17,)

                # Normalize using same function as during training
                try:
                    nkp = _normalize(keypoints, kp_conf)
                except Exception:
                    nkp = keypoints

                flattened = nkp.flatten().tolist()

                # Sequence buffer
                cam["seq"].append(flattened)
                if len(cam["seq"]) < SEQUENCE_LENGTH:
                    # Hali sequence to'lmagan
                    draw_pose(display_frame, keypoints, "loading")
                    cv2.putText(display_frame, 
                               f"Loading... {len(cam['seq'])}/{SEQUENCE_LENGTH}", 
                               (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    cv2.imshow(cam["id"], display_frame)
                    continue
                
                if len(cam["seq"]) > SEQUENCE_LENGTH:
                    cam["seq"].pop(0)

                seq_np = np.array(cam["seq"])  # [frames, 34]
                seq_tensor = torch.tensor(seq_np, dtype=torch.float32).T.unsqueeze(0).to(device)  # [1, 34, frames]

                with torch.no_grad():
                    output = tcn(seq_tensor)
                    probs = torch.softmax(output, dim=1)[0]  # [3] probabilities
                    pred = torch.argmax(probs).item()
                    
                    # ✅ Confidence filtering (threshold)
                    confidence = probs[pred].item()
                    
                    # Agar confidence past bo'lsa, no_fall deb hisoblash
                    if pred == 2 and confidence < 0.7:  # fall uchun yuqori threshold
                        pred = 0
                        label = "no_fall"
                    elif pred == 1 and confidence < 0.5:  # pre_fall uchun o'rtacha
                        pred = 0
                        label = "no_fall"
                    else:
                        label = LABEL_MAP[pred]
                    
                    # Temporal smoothing (oxirgi 3 ta prediction)
                    if "pred_history" not in cam:
                        cam["pred_history"] = []
                    cam["pred_history"].append(pred)
                    if len(cam["pred_history"]) > 3:
                        cam["pred_history"].pop(0)
                    
                    # Majority voting
                    if len(cam["pred_history"]) >= 3:
                        from collections import Counter
                        most_common = Counter(cam["pred_history"]).most_common(1)[0][0]
                        label = LABEL_MAP[most_common]

                # Pose va bounding box chizish
                draw_pose(display_frame, keypoints, label)
                
                # Confidence ko'rsatish
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(display_frame, conf_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # FPS va info
                cv2.putText(display_frame, f"Camera: {cam['id']}", 
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(cam["id"], display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[Main] ⛔ Stopping all cameras...")
    finally:
        for cam in cameras:
            cam["reader"].stop()
        cv2.destroyAllWindows()
        print("[Main] ✅ All processes closed.")


if __name__ == "__main__":
    main()