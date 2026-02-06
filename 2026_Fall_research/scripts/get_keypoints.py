import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# 1. Skript turgan joydan 2 ta papka tepaga chiqamiz (fall_research papkasiga)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CSV_PATH = os.path.join(BASE_DIR, "frame_labels_all.csv") 

print(f"Loyiha ildiz papkasi (Root): {BASE_DIR}")
print(f"CSV fayl yo'li: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print("!!! XATO: CSV fayli topilmadi!")
    exit()

labels_df = pd.read_csv(CSV_PATH)
model = YOLO('yolo11n-pose.pt')

def process_data_full():
    all_data = []
    found_count = 0

    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
        # CSV ichidagi yo'lni tozalash
        img_rel_path = row['filename'].strip().replace('\\', '/')
        
        # To'liq yo'lni hosil qilish
        img_abs_path = os.path.join(BASE_DIR, img_rel_path)
        
        # AGAR TOPILMASA: 'data/' so'zi takrorlanayotganini tekshiramiz
        if not os.path.exists(img_abs_path):
            # Ba'zan BASE_DIR ichida 'data' bor, rel_path ham 'data' bilan boshlanadi
            if "data/data" in img_abs_path.replace('\\', '/'):
                 img_abs_path = os.path.join(BASE_DIR, img_rel_path.replace('data/', '', 1))

        if not os.path.exists(img_abs_path):
            if index < 1: # Faqat birinchi xatoni ko'rsatish
                print(f"\nHali ham topilmadi. Qidirilgan manzil: {img_abs_path}")
            continue
        
        found_count += 1
        # Inference
        results = model(img_abs_path, verbose=False, conf=0.2)
        
        if len(results[0].keypoints) > 0 and results[0].keypoints.xyn is not None:
            kp = results[0].keypoints.xyn[0].cpu().numpy()
            if kp.shape[0] == 17:
                # 17 ta nuqtani (x, y) tekislab 34 ta qiymat qilamiz
                final_row = list(kp.flatten()) 
                # Binary label
                actual_label = 1 if row['label_id'] > 0 else 0 
                activity_id = f"{row['subject']}_{row['activity']}_{row['clip']}"
                all_data.append(final_row + [activity_id, actual_label])
            
    print(f"\nNatija: {found_count} ta rasm topildi va ishlandi.")
    return all_data

# Ma'lumotlarni yig'ishni boshlash
res = process_data_full()

if len(res) > 0:
    # 34 ta koordinata + activity + label
    columns = [f'kp_{i}' for i in range(34)] + ['activity', 'label']
    df = pd.DataFrame(res, columns=columns)
    
    # Natijani saqlash (skript bilan bir xil papkaga yoki bir tepaga)
    save_path = os.path.join(os.path.dirname(__file__), "upfall_full_kp_2026.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ MUVAFFAQIYATLI! {len(df)} ta kadr {save_path} fayliga saqlandi.")
else:
    print("❌ XATO: Hech qanday ma'lumot yig'ilmadi.")