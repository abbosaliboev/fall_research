"""
Full 2-stage fall detection training pipeline.

Stage 1 : ST-GCN  — trained end-to-end
Stage 2 : Physics Rule Filter — thresholds tuned on validation set

Results printed at end:
  - Stage 1 alone (ST-GCN)
  - Stage 1 + Stage 2 (ST-GCN + Physics)
  - Side-by-side comparison table
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

from stgcn import STGCN, PhysicsFilter, TwoStageDetector

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "cv_dataset")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "checkpoints")
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.5
FPS         = 19.0          # UP-Fall camera ~19 Hz
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CKPT_DIR, exist_ok=True)


# ── dataset ───────────────────────────────────────────────────────────────────

class FallDataset(Dataset):
    """(N, T, V, C) -> yields (C, T, V, M=1) tensor + label."""

    def __init__(self, X, y, augment=False):
        # (N, T, V, C) -> (N, C, T, V)
        self.X = torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                             # (C, T, V)
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)                      # horizontal mirror
            x = x + torch.randn_like(x) * 0.01     # small noise
        return x.unsqueeze(-1), self.y[idx]         # (C, T, V, 1)


def make_sampler(y):
    _, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    sw = torch.tensor([weights[c] for c in y], dtype=torch.float)
    return WeightedRandomSampler(sw, len(sw))


# ── training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(1)
        loss_sum += loss.item() * len(y)
        correct  += (preds == y).sum().item()
        total    += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (
        loss_sum / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ── tensor builder (for two-stage batch inference) ────────────────────────────

def to_stgcn_tensor(X_np: np.ndarray) -> torch.Tensor:
    """(N, T, V, C) -> (N, C, T, V, 1) tensor"""
    t = torch.from_numpy(X_np.transpose(0, 3, 1, 2)).float()  # (N, C, T, V)
    return t.unsqueeze(-1)                                      # (N, C, T, V, 1)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_results(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Fall F1  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['NO-FALL','FALL'])}")
    print(f"Confusion Matrix:\n{cm}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    X = np.load(os.path.join(DATA_DIR, "X.npy"))   # (N, T, V, C)
    y = np.load(os.path.join(DATA_DIR, "y.npy"))   # (N,)
    print(f"Data: X={X.shape}  y={y.shape}  FALL={y.sum()}  NO-FALL={(y==0).sum()}")

    # stratified split  70/15/15
    X_tr,  X_tmp, y_tr,  y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te,  y_val, y_te  = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    print(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)}\n")

    train_ds = FallDataset(X_tr,  y_tr,  augment=True)
    val_ds   = FallDataset(X_val, y_val)
    test_ds  = FallDataset(X_te,  y_te)

    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=make_sampler(y_tr))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False)

    # ── Stage 1: train ST-GCN ─────────────────────────────────────────────────
    print("=" * 50)
    print("  Stage 1: Training ST-GCN")
    print("=" * 50)

    model     = STGCN(in_channels=3, num_classes=2, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_preds, va_labels, _ = evaluate(model, val_loader, criterion)
        va_f1 = f1_score(va_labels, va_preds, pos_label=1, zero_division=0)
        scheduler.step()

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_stgcn.pth"))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                f"val_acc={va_acc:.3f} val_fall_f1={va_f1:.3f}  "
                f"best_f1={best_val_f1:.3f}"
            )

    # load best model
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_stgcn.pth")))
    print(f"\nLoaded best model (val fall F1={best_val_f1:.4f})")

    # ── Stage 1 test evaluation ───────────────────────────────────────────────
    _, _, s1_te_preds, y_te_labels, s1_te_probs = evaluate(model, test_loader, criterion)
    print_results("Stage 1 Only — ST-GCN", y_te_labels, s1_te_preds)

    # ── Stage 2: fit physics filter ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Stage 2: Fitting Physics Filter")
    print("=" * 50)

    physics = PhysicsFilter(fps=FPS)

    # get Stage-1 predictions on validation set for grid search
    _, _, s1_val_preds, _, s1_val_probs = evaluate(model, val_loader, criterion)

    # grid-search physics thresholds on validation
    physics.search_thresholds(X_val, y_val, s1_val_preds)

    # ── Two-stage: grid-search both thresholds on validation ─────────────────
    detector = TwoStageDetector(model, physics, device=DEVICE)

    X_val_t = to_stgcn_tensor(X_val).to(DEVICE)
    detector.tune_thresholds(X_val_t, X_val, y_val)

    # ── Two-stage test evaluation ─────────────────────────────────────────────
    X_te_t = to_stgcn_tensor(X_te).to(DEVICE)
    two_stage_preds = detector.predict_batch(X_te_t, X_te)
    print_results("Stage 1 + Stage 2 — ST-GCN + Physics (Rescue)", y_te_labels, two_stage_preds)

    # ── side-by-side summary ──────────────────────────────────────────────────
    s1_acc = accuracy_score(y_te_labels, s1_te_preds)
    s2_acc = accuracy_score(y_te_labels, two_stage_preds)
    s1_f1  = f1_score(y_te_labels, s1_te_preds,    pos_label=1, zero_division=0)
    s2_f1  = f1_score(y_te_labels, two_stage_preds, pos_label=1, zero_division=0)

    print("\n" + "=" * 50)
    print("  COMPARISON SUMMARY")
    print("=" * 50)
    print(f"  {'Model':<30} {'Accuracy':>10} {'Fall F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'ST-GCN (Stage 1)':<30} {s1_acc:>10.4f} {s1_f1:>10.4f}")
    print(f"  {'ST-GCN + Physics (2-Stage)':<30} {s2_acc:>10.4f} {s2_f1:>10.4f}")

    # save two-stage detector thresholds
    import json
    cfg = {
        "stage1_threshold":  detector.stage1_threshold,
        "rescue_threshold":  detector.rescue_threshold,
        "vel_threshold":     physics.vel_threshold,
        "acc_threshold":     physics.acc_threshold,
        "fps":               FPS,
    }
    with open(os.path.join(CKPT_DIR, "two_stage_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nConfig saved: {os.path.join(CKPT_DIR, 'two_stage_config.json')}")


if __name__ == "__main__":
    main()
