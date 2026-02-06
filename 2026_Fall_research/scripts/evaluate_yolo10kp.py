import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset import FallDataset # Dataset klassingizda 10-KP logic borligiga ishonch hosil qiling
from models import TCN_Attention_Model
import os

# 1. Sozlamalar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 10-KP uchun tozalangan CSV fayl yo'li
CSV_PATH = os.path.join("..", "upfall_clean_2026.csv") 
MODEL_PATH = os.path.join("..", "fall_detection_v4.pth")

# 2. Ma'lumotlarni yuklash
# Dataset klassingiz 10-KP formatida (20 features) ekanligini tekshiring
dataset = FallDataset(CSV_PATH)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. Modelni yuklash (input_size=20 bo'lishi shart)
model = TCN_Attention_Model(input_size=20).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

print(f"Model baholanmoqda ({DEVICE} yordamida)...")

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # Inference
        output = model(data)
        preds = torch.argmax(output, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

# 4. Metrikalarni hisoblash
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, 
                               target_names=['Normal', 'Fall'], 
                               digits=4) # Maqola uchun aniqroq raqamlar

print("\n10-KP Classification Report:\n")
print(report)

# 5. Confusion Matrix vizualizatsiyasi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # 10-KP uchun rangni o'zgartirdik (farqlash uchun)
            xticklabels=['Normal', 'Fall'], 
            yticklabels=['Normal', 'Fall'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Fall Detection Confusion Matrix (Optimized 10-KP Model)')

# Rasmni saqlash
output_image = '../confusion_matrix_10kp_final.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.show()

print(f"Natija '{output_image}' sifatida saqlandi.")