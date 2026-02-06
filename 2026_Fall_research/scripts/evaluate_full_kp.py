import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from models_full_kp import TCN_Attention_Model_Full
from tqdm import tqdm

# 1. Dataset klassi (Training kodidagi bilan bir xil bo'lishi shart)
class FallDatasetFull(Dataset):
    def __init__(self, csv_file, window_size=30):
        df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.features = df.iloc[:, :34].values.astype('float32')
        self.labels = df.iloc[:, -1].values.astype('long')
        
        self.valid_indices = range(len(self.labels) - window_size)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.window_size
        x_window = self.features[start:end]
        y_label = self.labels[end - 1]
        return torch.tensor(x_window), torch.tensor(y_label)

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baholash qurilmasi: {DEVICE}")

    # 2. Modelni yuklash
    model = TCN_Attention_Model_Full(input_size=34).to(DEVICE)
    model.load_state_dict(torch.load("fall_detection_full_kp.pth", map_location=DEVICE))
    model.eval()

    # 3. Test ma'lumotlarini yuklash
    # Eslatma: Agar testing uchun alohida CSV bo'lsa, nomini o'zgartiring
    dataset = FallDatasetFull("upfall_full_kp_2026.csv") 
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    print("Modelni baholash boshlandi...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            batch_x = batch_x.to(DEVICE)
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 4. Metrikalarni hisoblash
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nUmumiy aniqlik (Accuracy): {acc*100:.2f}%")
    print("\nKlassifikatsiya hisoboti:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Fall']))

    # 5. Confusion Matrix (Xatoliklar matritsasi)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fall'], 
                yticklabels=['Normal', 'Fall'])
    plt.xlabel('Predicted)')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Full-KP Model')
    
    # Rasm qilib saqlash
    plt.savefig('confusion_matrix_full_kp.png')
    print("Confusion Matrix 'confusion_matrix_full_kp.png' nomi bilan saqlandi.")
    plt.show()

if __name__ == "__main__":
    evaluate()