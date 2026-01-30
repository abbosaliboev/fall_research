import cv2
import torch
import numpy as np
from ultralytics import YOLO
from models import TCN_Attention_Model
import collections

# 1. Yuklash
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('yolo11n-pose.pt')

# Bizning modelni yuklash
model = TCN_Attention_Model().to(DEVICE)
model.load_state_dict(torch.load("../fall_detection_v3.pth", map_location=DEVICE))
model.eval()

# 2. Parametrlar
WANTED_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]
frame_buffer = collections.deque(maxlen=30) # Oxirgi 30 ta kadrni saqlash uchun

video_path = "../test_video.mov" # Videongiz yo'li
cap = cv2.VideoCapture(video_path)

print("Inference boshlandi...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # YOLO orqali keypoint olish
    results = yolo_model(frame, verbose=False)
    
    if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
        kp = results[0].keypoints.xyn[0].cpu().numpy()
        
        if len(kp) >= 17:
            # Nuqtalarni tayyorlash (10 ta nuqta)
            filtered_kp = kp[WANTED_KP]
            neck_x = (kp[5][0] + kp[6][0]) / 2
            neck_y = (kp[5][1] + kp[6][1]) / 2
            final_row = list(filtered_kp.flatten()) + [neck_x, neck_y]
            
            frame_buffer.append(final_row)
            
            # Agar 30 ta kadr yig'ilgan bo'lsa, TCN dan so'raymiz
            if len(frame_buffer) == 30:
                input_data = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    output = model(input_data)
                    prediction = torch.softmax(output, dim=1)
                    prob, class_idx = torch.max(prediction, dim=1)
                
                label = "FALLING!" if class_idx.item() == 1 and prob.item() > 0.7 else "Normal"
                color = (0, 0, 255) if label == "FALLING!" else (0, 255, 0)
                
                # Ekranga chiqarish
                cv2.putText(frame, f"{label} ({prob.item():.2f})", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Skeletni chizish (YOLO natijasi)
    annotated_frame = results[0].plot()
    cv2.imshow("Fall Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()