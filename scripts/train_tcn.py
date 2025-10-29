
import os
import re
import math
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
SAVE_DIR  = os.path.join(PROJECT_ROOT, "experiments", "FD-01")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
BATCH  = 128
LR     = 3e-4
WD     = 1e-4
DROPOUT= 0.2
CLASSES= 3
PATIENCE= 6
GRAD_CLIP = 1.0  # ixtiyoriy grad clipping

# Windows uchun xavfsiz: DataLoader'larda worker=0
NUM_WORKERS = 0

# ================== DATASET ==================
class SequenceCSVDataset(Dataset):
    """
    CSV format:
      subject, activity, clip, start_frame, end_frame, center_frame,
      label, label_id, seq_len, feat_dim, f0..fN
    """
    def __init__(self, csv_path: str):
        assert os.path.exists(csv_path), f"Not found: {csv_path}"
        self.df = pd.read_csv(csv_path)

        # faqat f0, f1, ... ustunlari
        feat_cols = [c for c in self.df.columns if re.fullmatch(r"f\d+", c)]
        if not feat_cols:
            raise ValueError("Feature ustunlar topilmadi (f0, f1, ...). CSV formatini tekshiring.")
        feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))

        if "label_id" not in self.df.columns:
            raise ValueError("CSVda 'label_id' ustuni yo'q.")

        if "seq_len" not in self.df.columns or "feat_dim" not in self.df.columns:
            raise ValueError("CSVda 'seq_len' va 'feat_dim' ustunlari bo‘lishi shart.")

        self.seq_len = int(self.df["seq_len"].iloc[0])
        self.feat_dim = int(self.df["feat_dim"].iloc[0])

        expected = self.seq_len * self.feat_dim
        if len(feat_cols) != expected:
            raise ValueError(
                f"Feature ustun soni mos emas: {len(feat_cols)} != seq_len*feat_dim ({expected})."
            )

        self.feat_cols = feat_cols
        self.X = self.df[self.feat_cols].values.astype("float32")
        self.y = self.df["label_id"].values.astype("int64")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx]  # [seq_len*feat_dim]
        # [seq_len, feat_dim] -> [C=feat_dim, L=seq_len] (TCN uchun)
        x = x.reshape(self.seq_len, self.feat_dim).T
        x = torch.from_numpy(x)  # float32
        y = torch.tensor(self.y[idx])  # int64
        return x, y


# ================== MODEL ==================
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        # SAME-length padding: pad = ((k-1)*dil)//2  (k=3 -> pad=dil)
        pad = ((k - 1) * dil) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in,  c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        # net(x) va proj(x) uzunligi bir xil bo'ladi (SAME padding)
        return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, in_ch, num_classes=3, drop=0.2):
        super().__init__()
        dilations = [1, 2, 4, 8]
        channels  = [64, 128, 256, 256]  # har qatlam chiqish kanali
        layers = []
        c = in_ch
        for out_c, d in zip(channels, dilations):
            layers.append(TCNBlock(c, out_c, k=3, dil=d, drop=drop))
            c = out_c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, num_classes)
        )

    def forward(self, x):  # x: [B, C=in_ch, L]
        y = self.tcn(x)    # [B, C, L]
        y = self.head(y)   # [B, num_classes]
        return y


# ================== UTILS ==================
def make_loader(csv_path, batch, weighted=False):
    ds = SequenceCSVDataset(csv_path)
    if weighted:
        labels = ds.y
        counts = np.bincount(labels, minlength=CLASSES)
        weights = 1.0 / (counts + 1e-6)
        sample_w = weights[labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            replacement=True
        )
        loader = DataLoader(
            ds, batch_size=batch, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
        )
    else:
        loader = DataLoader(
            ds, batch_size=batch, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
        )
    return ds, loader

def evaluate(model, loader, device, classes=3):
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

    # Macro-F1
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for gt, pr in zip(gts, preds):
        if gt == pr:
            per[gt]["tp"] += 1
        else:
            per[pr]["fp"] += 1
            per[gt]["fn"] += 1

    f1s = []
    for c in range(classes):
        tp, fp, fn = per[c]["tp"], per[c]["fp"], per[c]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1


# ================== TRAIN LOOP ==================
def main():
    print("Device:", DEVICE)
    assert os.path.exists(TRAIN_CSV), f"Not found: {TRAIN_CSV}"
    assert os.path.exists(VAL_CSV),   f"Not found: {VAL_CSV}"

    # Loaders
    train_ds, train_loader = make_loader(TRAIN_CSV, BATCH, weighted=True)
    val_ds,   val_loader   = make_loader(VAL_CSV,   BATCH, weighted=False)

    # Input channel = feat_dim (C), L = seq_len
    tmp_x, _ = next(iter(train_loader))
    in_ch = tmp_x.shape[1]
    seq_len = tmp_x.shape[2]

    # Print class counts (train)
    counts = np.bincount(train_ds.y, minlength=CLASSES)
    print("Train class counts:", counts.tolist())

    # Model
    model = TCN(in_ch=in_ch, num_classes=CLASSES, drop=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )
    crit = nn.CrossEntropyLoss()

    # AMP (yangi API)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

    torch.backends.cudnn.benchmark = True
    best_f1, wait = -1.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
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
            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)

        # Eval
        acc, f1 = evaluate(model, val_loader, DEVICE, classes=CLASSES)
        print(f"[{epoch:02d}] loss={train_loss:.4f}  val_acc={acc:.4f}  val_macroF1={f1:.4f}  lr={sch.get_last_lr()[0]:.2e}")

        # Save best
        if f1 > best_f1:
            best_f1, wait = f1, 0
            ckpt_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save({"model": model.state_dict(), "in_ch": in_ch}, ckpt_path)
            print("  ↳ best.pt updated")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    print("Done. Best Macro-F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
