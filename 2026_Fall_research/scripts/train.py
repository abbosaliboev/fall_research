import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from dataset import FallDataset
from models import TCN_Attention_Model

CSV_PATH = os.path.join("..", "upfall_clean_2026.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0005 # Learning rate biroz pasaytirildi

full_dataset = FallDataset(CSV_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TCN_Attention_Model().to(DEVICE)

# MUHIM: Class vaznlari. Fall (1) ga 10 baravar ko'proq vazn beramiz
weights = torch.tensor([1.0, 10.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Haqiqiy o'qitish boshlandi ({DEVICE})...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_acc = 100. * correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val_Acc={val_acc:.2f}%")

torch.save(model.state_dict(), os.path.join("..", "fall_detection_v3.pth"))
print("Yangi model saqlandi!")