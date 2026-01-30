import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# 1. Label ma'lumotlarini yuklash
labels_df = pd.read_csv('../../frame_labels_all.csv') # Yo'lni to'g'rilang
# Filenameni key, label_id ni value qilib lug'at (dictionary) yaratamiz
label_map = dict(zip(labels_df['filename'], labels_df['label_id']))

model = YOLO('yolo11n-pose.pt')
WANTED_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]

def process_data():
    all_data = []
    # labels_df dagi barcha rasmlar bo'ylab yuramiz
    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Keypoints olinmoqda"):
        img_rel_path = row['filename'] # "data\fall_data\..."
        img_abs_path = os.path.join("..", "..", img_rel_path) # Absolute path yasash
        
        if not os.path.exists(img_abs_path):
            continue

        results = model(img_abs_path, verbose=False)
        
        if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
            kp = results[0].keypoints.xyn[0].cpu().numpy()
            if len(kp) < 17: continue
            
            filtered_kp = kp[WANTED_KP]
            neck_x = (kp[5][0] + kp[6][0]) / 2
            neck_y = (kp[5][1] + kp[6][1]) / 2
            final_row = list(filtered_kp.flatten()) + [neck_x, neck_y]
            
            # Labelni CSV dan olamiz (0: no_fall, 1: pre_fall, 2: fall bo'lishi mumkin)
            # Lekin bizga Binary classification (0 yoki 1) kerak bo'lsa:
            actual_label = 1 if row['label_id'] > 0 else 0 
            
            # Activity nomi (oynalash uchun kerak)
            activity_id = f"{row['subject']}_{row['activity']}_{row['clip']}"
            
            all_data.append(final_row + [activity_id, actual_label])
            
    return all_data

# Saqlash qismi... (oldingi koddagidek)
res = process_data()
df = pd.DataFrame(res, columns=[f'kp_{i}' for i in range(20)] + ['activity', 'label'])
df.to_csv('../upfall_clean_2026.csv', index=False)