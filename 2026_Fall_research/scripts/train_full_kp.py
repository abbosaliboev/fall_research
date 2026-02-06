import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models_full_kp import TCN_Attention_Model_Full
from tqdm import tqdm

# 1. Dataset klassi (30 kadrli oynani shakllantiradi)
class FallDatasetFull(Dataset):
    def __init__(self, csv_file, window_size=30):
        df = pd.read_csv(csv_file)
        self.window_size = window_size
        
        # Xususiyatlar (34 ta koordinata) va Labellar
        self.features = df.iloc[:, :34].values.astype('float32')
        self.labels = df.iloc[:, -1].values.astype('long')
        
        # Faqat to'liq 30 talik oyna bo'la oladigan indekslarni olamiz
        self.valid_indices = []
        for i in range(len(self.labels) - window_size):
            # Oyna ichidagi barcha kadrlar bir xil activity (video) ga tegishliligini tekshirish
            # (Agar CSVda 'activity' ustuni bo'lsa)
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.window_size
        
        # [30, 34] shaklidagi ma'lumot
        x_window = self.features[start:end]
        # Oynaning oxirgi kadridagi labelni olamiz
        y_label = self.labels[end - 1]
        
        return torch.tensor(x_window), torch.tensor(y_label)

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ishlatilayotgan qurilma: {DEVICE}")

    # 2. Ma'lumotlarni yuklash
    dataset = FallDatasetFull("upfall_full_kp_2026.csv")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Model, Loss va Optimizer
    model = TCN_Attention_Model_Full(input_size=34).to(DEVICE)
    # Klasslar balansi uchun (yiqilish kam bo'lsa, vaznni oshiramiz)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # 4. Training Loop
    epochs = 50
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=False)
        for batch_x, batch_y in loop:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE).long() 

            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistika
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        print(f"Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")

    # 5. Saqlash
    torch.save(model.state_dict(), "fall_detection_full_kp.pth")
    print("Model 'fall_detection_full_kp.pth' nomi bilan saqlandi!")

if __name__ == "__main__":
    train()