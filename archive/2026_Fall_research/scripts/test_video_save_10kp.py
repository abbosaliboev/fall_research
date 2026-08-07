import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from models import TCN_Attention_Model 

# 1. Qurilmani aniqlash va modellarni yuklash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ishlatilayotgan qurilma: {device}")

pose_model = YOLO('yolo11n-10kp.pt') 

tcn_model = TCN_Attention_Model(input_size=20) 
state_dict = torch.load('../fall_detection_v4.pth', map_location=device)
tcn_model.load_state_dict(state_dict)
tcn_model.to(device)
tcn_model.eval()

# 2. Videoni ochish
input_video = '../test_video.mov'
cap = cv2.VideoCapture(input_video)
w, h, orig_fps = (int(cap.get(x)) for x in (3, 4, 5))
out = cv2.VideoWriter('result_10kp_tcn.mp4', cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (w, h))

window_size = 30
sequence = []
total_frames = 0
start_process_time = time.time() # Umumiy vaqt boshi

print("Ishlov berilmoqda...")

while cap.isOpened():
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret: break

    results = pose_model(frame, verbose=False)
    total_frames += 1
    
    current_person_detected = False

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            current_person_detected = True
            
            # Keypointlarni chizish
            kpts_pixel = r.keypoints.xy.cpu().numpy()[0]
            for kp in kpts_pixel:
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

            # TCN tahlili
            if len(r.keypoints.xyn) > 0:
                kpts = r.keypoints.xyn.cpu().numpy()[0].flatten()
                if len(kpts) == 20:
                    sequence.append(kpts)
                
                if len(sequence) > window_size:
                    sequence.pop(0)

                if len(sequence) == window_size:
                    input_data = torch.FloatTensor(np.array([sequence])).to(device)
                    with torch.no_grad():
                        output = tcn_model(input_data)
                        prediction = torch.softmax(output, dim=1)
                        prob = torch.max(prediction).item()
                        is_fall = torch.argmax(prediction).item()

                    label, color = (f"FALL! {prob:.2f}", (0, 0, 255)) if is_fall == 1 and prob > 0.85 else (f"Normal {prob:.2f}", (0, 255, 0))
                    if is_fall == 1 and prob > 0.85: cv2.rectangle(frame, (0,0), (w,h), color, 10)
                    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if not current_person_detected:
        sequence = []

    # Instant FPS
    curr_fps = 1.0 / (time.time() - frame_start)
    cv2.putText(frame, f"FPS: {curr_fps:.1f}", (w-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Natijaviy videoni yozish(o'chirsa fps tezlashadi)
    out.write(frame)
    cv2.imshow('10KP + TCN Fall Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 3. Yakuniy statistikani hisoblash
end_process_time = time.time()
total_time = end_process_time - start_process_time
avg_fps = total_frames / total_time

print("\n" + "="*30)
print(f"Jarayon yakunlandi!")
print(f"Jami kadrlar: {total_frames}")
print(f"Umumiy vaqt: {total_time:.2f} sekund")
print(f"O'rtacha (AVG) FPS: {avg_fps:.2f}") # Mana siz so'ragan natija
print("="*30)

cap.release()
out.release()
cv2.destroyAllWindows()