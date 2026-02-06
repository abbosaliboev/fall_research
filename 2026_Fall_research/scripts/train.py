# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from dataset import FallDataset
from models import TCN_Attention_Model

# 1. Sozlamalar
CSV_PATH = os.path.join("..", "upfall_clean_2026.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0005

# 2. Datasetni yuklash
full_dataset = FallDataset(CSV_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model va Loss
model = TCN_Attention_Model().to(DEVICE)
weights = torch.tensor([1.0, 4.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Grafik uchun ma'lumotlarni yig'ish
history = {'train_loss': [], 'val_acc': []}

print(f"O'qitish boshlandi ({DEVICE})...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_acc = 100. * correct / len(val_loader.dataset)
    
    # Tarixni saqlash
    history['train_loss'].append(avg_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.2f}%")

# 4. Modelni saqlash
torch.save(model.state_dict(), os.path.join("..", "fall_detection_v5.pth"))
print("Model saqlandi!")

# 5. GRAFIK CHIZISH
plt.figure(figsize=(12, 5))

# Loss grafigi
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), history['train_loss'], 'r-', label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy grafigi
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), history['val_acc'], 'b-', label='Val Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('../training_results_plot.png') # Rasm bo'lib saqlanadi
plt.show() # Ekranda ko'rsatish
print("Grafik 'training_results_plot.png' sifatida saqlandi.")