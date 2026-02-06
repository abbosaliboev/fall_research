import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# 1. Papka manzillarini skrinshatga qarab sozlaymiz
# Skript 'scripts' papkasida turibdi, shuning uchun bir marta tepaga chiqamiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "frame_labels_all.csv") 

print(f"Loyiha papkasi: {BASE_DIR}")
print(f"CSV manzili: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print("!!! XATO: frame_labels_all.csv topilmadi!")
    exit()

labels_df = pd.read_csv(CSV_PATH)
model = YOLO('yolo11n-pose.pt')

def process_data_full():
    all_data = []
    found_count = 0

    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
        # CSV ichidagi 'filename' ustunini olamiz (masalan: data/fall_data/...)
        img_rel_path = row['filename'].strip().replace('\\', '/')
        
        # To'liq yo'l: BASE_DIR + data/fall_data/...
        img_abs_path = os.path.normpath(os.path.join(BASE_DIR, img_rel_path))
        
        # Birinchi rasmda yo'lni tekshirib ko'rsatamiz
        if index == 0:
            print(f"\nTekshirilayotgan rasm: {img_abs_path}")
            if os.path.exists(img_abs_path):
                print("✅ Rasm topildi!")
            else:
                print("❌ Rasm hali ham topilmadi. Yo'lni qayta tekshiring.")

        if not os.path.exists(img_abs_path):
            continue
        
        found_count += 1
        # Inference
        results = model(img_abs_path, verbose=False, conf=0.2)
        
        if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
            kp = results[0].keypoints.xyn[0].cpu().numpy()
            if kp.shape[0] == 17:
                # 34 ta koordinata (17*2)
                final_row = list(kp.flatten()) 
                # Label (0 yoki 1)
                actual_label = 1 if row['label_id'] > 0 else 0 
                # Activity identifikatori
                activity_id = f"{row['subject']}_{row['activity']}_{row['clip']}"
                
                all_data.append(final_row + [activity_id, actual_label])
            
    print(f"\nNatija: {found_count} ta rasm muvaffaqiyatli o'qildi.")
    return all_data

# Ijro
res = process_data_full()

if len(res) > 0:
    columns = [f'kp_{i}' for i in range(34)] + ['activity', 'label']
    df = pd.DataFrame(res, columns=columns)
    
    # Natijani loyiha ildiziga saqlaymiz
    save_path = os.path.join(BASE_DIR, "upfall_full_kp_2026.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ DATASET TAYYOR: {save_path}")
else:
    print("❌ XATO: Hech qanday ma'lumot saqlanmadi.")