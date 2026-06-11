"""
ST-GCN training script for fall detection (UP-Fall dataset).

Data:   cv_dataset/X.npy  (N, T, V, C)
        cv_dataset/y.npy  (N,)

ST-GCN expects (N, C, T, V, M) — this script handles the conversion.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from stgcn import STGCN

# ── config ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "cv_dataset")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "checkpoints")
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)


class FallDataset(Dataset):
    def __init__(self, X, y, augment=False):
        # X: (N, T, V, C)  ->  store as (N, C, T, V)
        self.X = torch.from_numpy(X.transpose(0, 3, 1, 2)).float()  # (N, C, T, V)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]          # (C, T, V)
        if self.augment:
            # random horizontal flip (mirror left-right joints)
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            # small gaussian noise
            x = x + torch.randn_like(x) * 0.01
        # add person dimension M=1 -> (C, T, V, 1)
        x = x.unsqueeze(-1)
        # ST-GCN wants (C, T, V, M) which we permute to (C, T, V, M)
        # final shape returned: (C, T, V, M=1)
        return x, self.y[idx]


def make_sampler(y_train):
    """WeightedRandomSampler to handle class imbalance."""
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[c] for c in y_train], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # x: (N, C, T, V, M)  — already (N, C, T, V, 1) from dataset
        x = x.permute(0, 1, 2, 3, 4)   # no-op, already correct shape
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    print(f"Device: {DEVICE}")

    X = np.load(os.path.join(DATA_DIR, "X.npy"))  # (N, T, V, C)
    y = np.load(os.path.join(DATA_DIR, "y.npy"))  # (N,)
    print(f"Loaded X={X.shape}  y={y.shape}  FALL={y.sum()}  NO-FALL={(y==0).sum()}")

    # stratified split: 70% train, 15% val, 15% test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    print(f"Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")

    train_ds = FallDataset(X_tr,  y_tr,  augment=True)
    val_ds   = FallDataset(X_val, y_val, augment=False)
    test_ds  = FallDataset(X_te,  y_te,  augment=False)

    sampler     = make_sampler(y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = STGCN(in_channels=3, num_classes=2, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_stgcn.pth"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                  f"val loss={va_loss:.4f} acc={va_acc:.3f}  "
                  f"best_val={best_val_acc:.3f}")

    # ── final test evaluation ────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_stgcn.pth")))
    _, te_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["NO-FALL", "FALL"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
