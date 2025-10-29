# -*- coding: utf-8 -*-
# scripts/eval_tcn.py
import os, numpy as np, torch, torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tcn_dataset import SequenceCSVDataset

PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
TEST_CSV = os.path.join(PROJECT_ROOT, "sequences_test.csv")
CKPT     = os.path.join(PROJECT_ROOT, "experiments", "FD-01", "best.pt")
CLASSES  = 3
LABEL_NAMES = ["no_fall","pre_fall","fall"]

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        pad = (k-1)*dil
        self.net = nn.Sequential(
            nn.Conv1d(c_in,  c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()
    def forward(self, x): return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, in_ch, num_classes=3, drop=0.2):
        super().__init__()
        widths = [64,128,256]
        dilations = [1,2,4,8]
        layers=[]; c=in_ch
        for i,d in enumerate(dilations):
            layers.append(TCNBlock(c, widths[min(i,len(widths)-1)], k=3, dil=d, drop=drop))
            c = widths[min(i,len(widths)-1)]
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, num_classes))
    def forward(self, x):
        return self.head(self.tcn(x))

def main():
    assert os.path.exists(TEST_CSV), f"Not found: {TEST_CSV}"
    ck = torch.load(CKPT, map_location="cpu")
    in_ch = ck.get("in_ch", None)
    assert in_ch is not None, "Checkpointda in_ch topilmadi."

    ds = SequenceCSVDataset(TEST_CSV)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    model = TCN(in_ch=in_ch, num_classes=CLASSES, drop=0.2)
    model.load_state_dict(ck["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    all_p, all_y = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            p = logits.argmax(1)
            all_p.extend(p.cpu().numpy().tolist())
            all_y.extend(y.cpu().numpy().tolist())

    all_p = np.array(all_p); all_y = np.array(all_y)
    # Confusion matrix
    cm = np.zeros((CLASSES,CLASSES), dtype=int)
    for gt,pr in zip(all_y, all_p): cm[gt,pr]+=1

    # Per-class F1
    f1s=[]
    for c in range(CLASSES):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    acc = float((all_p==all_y).mean())

    print("Confusion matrix (rows=GT, cols=Pred):")
    df_cm = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print(df_cm)
    print("\nPer-class F1:", {LABEL_NAMES[i]: round(f1s[i],3) for i in range(CLASSES)})
    print("Macro-F1:", round(macro_f1,3), "  Acc:", round(acc,3))

    # ixtiyoriy: CSVga saqlash
    out_cm = os.path.join(PROJECT_ROOT, "experiments", "FD-01", "confusion_matrix_test.csv")
    df_cm.to_csv(out_cm, index=True)
    print("Saved:", out_cm)

if __name__ == "__main__":
    main()
