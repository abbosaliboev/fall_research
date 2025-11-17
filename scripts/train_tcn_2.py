# train_tcn_2.py
"""
TCN Training Script with:
- Weighted sampling for imbalanced classes
- Data augmentation for minority classes
- Mixed precision training (AMP)
- Early stopping
- Class-wise F1 metrics
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ================== PATH & CONFIG ==================
PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
TRAIN_CSV = os.path.join(PROJECT_ROOT, "sequences_train.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "sequences_val.csv")
SAVE_DIR  = os.path.join(PROJECT_ROOT, "experiments", "FD-02")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH  = 32
LR     = 1e-3
WD     = 1e-4
DROPOUT= 0.3
CLASSES= 3
PATIENCE= 15
GRAD_CLIP = 1.0
NUM_WORKERS = 0

LABEL_NAMES = {0: "no_fall", 1: "pre_fall", 2: "fall"}

# ================== DATASET ==================
class SequenceCSVDataset(Dataset):
    def __init__(self, csv_path: str, augment=False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV topilmadi: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        if len(self.df) == 0:
            raise ValueError(f"CSV bo'sh: {csv_path}")
        
        # Feature columns
        feat_cols = [c for c in self.df.columns if re.fullmatch(r"f\d+", c)]
        if not feat_cols:
            raise ValueError(f"Feature columns (f0, f1, ...) topilmadi: {csv_path}")
        feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))
        
        # Metadata
        if "seq_len" not in self.df.columns or "feat_dim" not in self.df.columns:
            raise ValueError(f"'seq_len' yoki 'feat_dim' ustunlari yo'q: {csv_path}")
        
        self.seq_len = int(self.df["seq_len"].iloc[0])
        self.feat_dim = int(self.df["feat_dim"].iloc[0])
        
        # Data
        self.X = self.df[feat_cols].values.astype("float32")
        
        if "label_id" not in self.df.columns:
            raise ValueError(f"'label_id' ustuni yo'q: {csv_path}")
        self.y = self.df["label_id"].values.astype("int64")
        
        self.augment = augment
        
        # Validation
        expected_feats = self.seq_len * self.feat_dim
        if len(feat_cols) != expected_feats:
            raise ValueError(
                f"Feature count mismatch: {len(feat_cols)} != {expected_feats} "
                f"(seq_len={self.seq_len}, feat_dim={self.feat_dim})"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(self.seq_len, self.feat_dim).T  # [feat_dim, seq_len]
        x = torch.from_numpy(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        # Augmentation faqat minority classes uchun
        if self.augment and y.item() != 0:
            # Gaussian noise
            x = x + 0.02 * torch.randn_like(x)
            # Temporal jittering (optional)
            if torch.rand(1) < 0.3:  # 30% chance
                shift = int(torch.randint(-2, 3, (1,)).item())
                x = torch.roll(x, shifts=shift, dims=1)
        
        return x, y

# ================== MODEL ==================
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        pad = ((k - 1) * dil) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, dilation=dil, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, k, dilation=dil, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        
    def forward(self, x):
        return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, in_ch, num_classes=3, drop=0.2):
        super().__init__()
        dilations = [1, 2, 4, 8]
        channels  = [64, 128, 256, 256]
        
        layers = []
        c = in_ch
        for out_c, d in zip(channels, dilations):
            layers.append(TCNBlock(c, out_c, k=3, dil=d, drop=drop))
            c = out_c
        
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(c, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.tcn(x))

# ================== UTILS ==================
def make_loader(csv_path, batch, weighted=False, augment=False, shuffle=True):
    """Create DataLoader with optional weighted sampling"""
    ds = SequenceCSVDataset(csv_path, augment=augment)
    
    print(f"[INFO] Loaded {os.path.basename(csv_path)}: {len(ds)} sequences")
    
    if weighted:
        # Weighted random sampler for imbalanced data
        labels = ds.y
        counts = np.bincount(labels, minlength=CLASSES)
        print(f"  Class distribution: {counts}")
        
        # Inverse frequency weights
        weights = 1.0 / (counts + 1e-6)
        sample_w = weights[labels]
        
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_w).double(), 
            num_samples=len(sample_w), 
            replacement=True
        )
        loader = DataLoader(
            ds, 
            batch_size=batch, 
            sampler=sampler, 
            num_workers=NUM_WORKERS, 
            pin_memory=(DEVICE=="cuda")
        )
    else:
        loader = DataLoader(
            ds, 
            batch_size=batch, 
            shuffle=shuffle, 
            num_workers=NUM_WORKERS, 
            pin_memory=(DEVICE=="cuda")
        )
    
    return ds, loader

def evaluate(model, loader, device, classes=3):
    """Evaluate model and compute per-class metrics"""
    model.eval()
    correct, total = 0, 0
    preds, gts = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = logits.argmax(1)
            preds.extend(p.cpu().numpy().tolist())
            gts.extend(y.cpu().numpy().tolist())
            correct += (p == y).sum().item()
            total += y.numel()
    
    acc = correct / max(1, total)
    
    # Per-class metrics
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for gt, pr in zip(gts, preds):
        if gt == pr:
            per_class[gt]["tp"] += 1
        else:
            per_class[pr]["fp"] += 1
            per_class[gt]["fn"] += 1
    
    f1s = []
    print("  Per-class metrics:")
    for c in range(classes):
        tp = per_class[c]["tp"]
        fp = per_class[c]["fp"]
        fn = per_class[c]["fn"]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        
        label_name = LABEL_NAMES.get(c, f"class_{c}")
        print(f"    {label_name:10s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1

# ================== TRAIN LOOP ==================
def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Training config: epochs={EPOCHS}, batch={BATCH}, lr={LR}")
    
    # Load data
    train_ds, train_loader = make_loader(TRAIN_CSV, BATCH, weighted=True, augment=True)
    val_ds, val_loader     = make_loader(VAL_CSV, BATCH, weighted=False, augment=False, shuffle=False)
    
    # Model setup
    tmp_x, _ = next(iter(train_loader))
    in_ch = tmp_x.shape[1]  # feat_dim
    seq_len = tmp_x.shape[2]
    
    print(f"[INFO] Input shape: [{in_ch}, {seq_len}]")
    
    model = TCN(in_ch=in_ch, num_classes=CLASSES, drop=DROPOUT).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    # Optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=LR, 
        epochs=EPOCHS, 
        steps_per_epoch=len(train_loader)
    )
    
    crit = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    
    best_f1, wait = -1.0, 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            
            if DEVICE == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                if GRAD_CLIP is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            
            sch.step()
            running_loss += loss.item() * y.size(0)
        
        train_loss = running_loss / len(train_ds)
        
        # Validation
        acc, f1 = evaluate(model, val_loader, DEVICE, classes=CLASSES)
        
        lr_current = sch.get_last_lr()[0]
        print(f"\n[Epoch {epoch:02d}/{EPOCHS}] loss={train_loss:.4f} | val_acc={acc:.4f} | macro_F1={f1:.4f} | lr={lr_current:.2e}")
        
        # Save best model
        if f1 > best_f1:
            best_f1, wait = f1, 0
            ckpt_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "in_ch": in_ch,
                "epoch": epoch,
                "f1": f1,
                "acc": acc
            }, ckpt_path)
            print(f"  ✅ best.pt updated (F1={f1:.4f})")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n⛔ Early stopping (no improvement for {PATIENCE} epochs)")
                break
    
    print("\n" + "="*60)
    print(f"✅ Training complete! Best Macro-F1: {best_f1:.4f}")
    print(f"✅ Model saved to: {os.path.join(SAVE_DIR, 'best.pt')}")
    print("="*60)

if __name__ == "__main__":
    main()