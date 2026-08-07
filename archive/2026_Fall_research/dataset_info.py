import pandas as pd
import os

def generate_dataset_report(csv_path):
    # CSV faylni o'qish
    if not os.path.exists(csv_path):
        print(f"❌ Xato: {csv_path} fayli topilmadi!")
        return

    df = pd.read_csv(csv_path)

    # Ma'lumotlarni hisoblash
    total_frames = len(df)
    unique_labels = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100
    
    unique_subjects = df['subject'].nunique()
    unique_activities = df['activity'].nunique()
    
    # Kliplar sonini aniqlash (Subject + Activity + Clip kombinatsiyasi bo'yicha)
    total_clips = df.groupby(['subject', 'activity', 'clip']).size().reset_index().shape[0]

    # Hisobotni chop etish
    print("\n" + "="*50)
    print("📊 DATASET ANALIZI VA TAHLILI HISOBOTI")
    print("="*50)
    print(f"📂 Fayl nomi:             {csv_path}")
    print(f"🖼️ Umumiy kadrlar soni:    {total_frames} ta")
    print(f"🎬 Umumiy kliplar soni:    {total_clips} ta")
    print(f"👤 Odamlar (Subject) soni: {unique_subjects} ta")
    print(f"🎭 Harakat turlari soni:   {unique_activities} ta")
    
    print("-" * 50)
    print("🏷️ KLASSLAR TAQSIMOTI (Labels):")
    label_report = pd.DataFrame({
        'Kadrlar soni': unique_labels,
        'Ulushi (%)': label_percentages.round(2)
    })
    print(label_report)
    
    print("-" * 50)
    print("📋 HAR BIR SUBJECT BO'YICHA MA'LUMOT:")
    subject_stats = df.groupby('subject').agg({
        'label': 'count',
        'activity': 'nunique',
        'clip': 'nunique'
    }).rename(columns={'label': 'Kadrlar', 'activity': 'Harakatlar', 'clip': 'Kliplar'})
    print(subject_stats)
    
    print("="*50 + "\n")

# Skriptni ishga tushirish
if __name__ == "__main__":
    FILE_NAME = "frame_labels_all.csv"  # Faylingiz nomi
    generate_dataset_report(FILE_NAME)