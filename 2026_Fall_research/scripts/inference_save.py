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

# 2. Keypoint sozlamalari
WANTED_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]
SKELETON_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

frame_buffer = collections.deque(maxlen=30)
fall_counter = 0
prev_frame_time = 0
FALL_THRESHOLD = 0.80

# --- FPS LOGLASH UCHUN O'ZGARUVCHILAR ---
total_start_time = time.time()
frame_count = 0
# -------------------------------------

video_path = "../test_video.mov"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS)) or 30

out = cv2.VideoWriter(
    "../fall_detection_final_10kp_2.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_input,
    (frame_width, frame_height)
)

print("Inference boshlandi...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    annotated_frame = frame.copy()
    tcn_time = 0
    conf = 0.0

    results = yolo_model(frame, verbose=False)

    if (
        len(results[0].keypoints) > 0
        and results[0].keypoints.xyn is not None
    ):
        kp_norm = results[0].keypoints.xyn[0].cpu().numpy()
        kp_pixel = results[0].keypoints.xy[0].cpu().numpy()

        if len(kp_norm) >= 17:
            neck_x = (kp_norm[5][0] + kp_norm[6][0]) / 2
            neck_y = (kp_norm[5][1] + kp_norm[6][1]) / 2

            neck_px = (
                int((kp_pixel[5][0] + kp_pixel[6][0]) / 2),
                int((kp_pixel[5][1] + kp_pixel[6][1]) / 2)
            )

            cv2.circle(annotated_frame, neck_px, 6, (0, 255, 255), -1)

            for idx in WANTED_KP:
                pos = (
                    int(kp_pixel[idx][0]),
                    int(kp_pixel[idx][1])
                )
                cv2.circle(annotated_frame, pos, 5, (255, 0, 0), -1)

            for start, end in SKELETON_CONNECTIONS:
                pt1 = (
                    int(kp_pixel[start][0]),
                    int(kp_pixel[start][1])
                )
                pt2 = (
                    int(kp_pixel[end][0]),
                    int(kp_pixel[end][1])
                )
                cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)

            filtered_kp = kp_norm[WANTED_KP].flatten()
            final_input = np.append(filtered_kp, [neck_x, neck_y])
            frame_buffer.append(final_input)

            if len(frame_buffer) == 30:
                input_tensor = torch.tensor(
                    [list(frame_buffer)],
                    dtype=torch.float32
                ).to(DEVICE)

                t_start = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf, class_idx = torch.max(prob, dim=1)

                tcn_time = (time.time() - t_start) * 1000
                conf = conf.item()

                if class_idx.item() == 1 and conf > FALL_THRESHOLD:
                    fall_counter = min(20, fall_counter + 2)
                else:
                    fall_counter = max(0, fall_counter - 1)

    is_falling = fall_counter >= 10
    status_text = "FALL DETECTED!" if is_falling else "Normal"
    status_color = (0, 0, 255) if is_falling else (0, 255, 0)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_time

    cv2.rectangle(annotated_frame, (10, 10), (450, 130), (0, 0, 0), -1)
    cv2.putText(
        annotated_frame,
        status_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3
    )

    cv2.putText(
        annotated_frame,
        f"Conf: {conf:.2f} | Counter: {fall_counter}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)} | TCN: {tcn_time:.1f}ms",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1
    )

    out.write(annotated_frame)
    cv2.imshow("Optimized 10-KP Fall Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- VIDEO TUGAGANDAN KEYIN FPS LOGLASH ---
total_end_time = time.time()
total_duration = total_end_time - total_start_time
avg_fps = frame_count / total_duration if total_duration > 0 else 0

print("\n" + "=" * 30)
print(f"Jami ishlangan kadrlar: {frame_count}")
print(f"Umumiy vaqt: {total_duration:.2f} soniya")
print(f"O'RTACHA FPS: {avg_fps:.2f}")
print("=" * 30)
# ---------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()
