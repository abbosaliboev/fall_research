# -*- coding: utf-8 -*-
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
EPOCHS = 40
BATCH  = 64  # batchni biroz kamaytirdik
LR     = 1e-4  # max LR ni pasaytirdik
WD     = 1e-4
DROPOUT= 0.2
CLASSES= 3
PATIENCE= 10
GRAD_CLIP = 1.0
NUM_WORKERS = 0

# ================== DATASET ==================
class SequenceCSVDataset(Dataset):
    def __init__(self, csv_path: str, augment=False):
        self.df = pd.read_csv(csv_path)
        feat_cols = [c for c in self.df.columns if re.fullmatch(r"f\d+", c)]
        feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))
        self.seq_len = int(self.df["seq_len"].iloc[0])
        self.feat_dim = int(self.df["feat_dim"].iloc[0])
        self.X = self.df[feat_cols].values.astype("float32")
        self.y = self.df["label_id"].values.astype("int64")
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(self.seq_len, self.feat_dim).T
        x = torch.from_numpy(x)
        y = torch.tensor(self.y[idx])
        if self.augment and y.item() != 0:  # minor class augmentation
            x = x + 0.01 * torch.randn_like(x)  # small Gaussian noise
        return x, y

# ================== MODEL ==================
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        pad = ((k - 1) * dil) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, in_ch, num_classes=3, drop=0.2):
        super().__init__()
        dilations = [1,2,4,8]
        channels  = [64,128,256,256]
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
    def forward(self, x):
        return self.head(self.tcn(x))

# ================== UTILS ==================
def make_loader(csv_path, batch, weighted=False, augment=False):
    ds = SequenceCSVDataset(csv_path, augment=augment)
    if weighted:
        labels = ds.y
        counts = np.bincount(labels, minlength=CLASSES)
        weights = 1.0 / (counts + 1e-6)
        sample_w = weights[labels]
        sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(), num_samples=len(sample_w), replacement=True)
        loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))
    else:
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))
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
            correct += (p==y).sum().item()
            total += y.numel()
    acc = correct / max(1,total)
    per = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    for gt, pr in zip(gts, preds):
        if gt==pr:
            per[gt]["tp"] += 1
        else:
            per[pr]["fp"] += 1
            per[gt]["fn"] += 1
    f1s=[]
    for c in range(classes):
        tp, fp, fn = per[c]["tp"], per[c]["fp"], per[c]["fn"]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    return acc, float(np.mean(f1s))

# ================== TRAIN LOOP ==================
def main():
    print("Device:", DEVICE)
    train_ds, train_loader = make_loader(TRAIN_CSV, BATCH, weighted=True, augment=True)
    val_ds, val_loader     = make_loader(VAL_CSV, BATCH, weighted=False, augment=False)
    
    tmp_x,_ = next(iter(train_loader))
    in_ch = tmp_x.shape[1]
    seq_len = tmp_x.shape[2]
    
    print("Train class counts:", np.bincount(train_ds.y, minlength=CLASSES).tolist())
    
    model = TCN(in_ch=in_ch, num_classes=CLASSES, drop=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if DEVICE=="cuda" else None
    torch.backends.cudnn.benchmark = True
    
    best_f1, wait = -1.0, 0
    for epoch in range(1,EPOCHS+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if DEVICE=="cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = crit(logits,y)
                scaler.scale(loss).backward()
                if GRAD_CLIP is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits,y)
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sch.step()
            running += loss.item()*y.size(0)
        train_loss = running/len(train_ds)
        acc,f1 = evaluate(model,val_loader,DEVICE,classes=CLASSES)
        print(f"[{epoch:02d}] loss={train_loss:.4f} val_acc={acc:.4f} val_macroF1={f1:.4f} lr={sch.get_last_lr()[0]:.2e}")
        if f1>best_f1:
            best_f1, wait = f1, 0
            ckpt_path = os.path.join(SAVE_DIR,"best.pt")
            torch.save({"model":model.state_dict(),"in_ch":in_ch}, ckpt_path)
            print("  ↳ best.pt updated")
        else:
            wait+=1
            if wait>=PATIENCE:
                print("Early stopping.")
                break
    print("Done. Best Macro-F1:", round(best_f1,4))

if __name__=="__main__":
    main()
