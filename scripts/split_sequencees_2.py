# -*- coding: utf-8 -*-
# scripts/split_sequences.py
import os, pandas as pd, numpy as np
from collections import Counter, defaultdict

PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
IN_CSV  = os.path.join(PROJECT_ROOT, "fall_sequences_priority_label.csv")
OUT_TRAIN = os.path.join(PROJECT_ROOT, "sequences_train.csv")
OUT_VAL   = os.path.join(PROJECT_ROOT, "sequences_val.csv")
OUT_TEST  = os.path.join(PROJECT_ROOT, "sequences_test.csv")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RNG = np.random.default_rng(42)
MIN_PER_SPLIT = 10  # har label uchun val/test minimal sequence

def stratified_split_sequence(df):
    """Har label uchun minimal val/test count saqlagan holda split qilamiz"""
    df = df.copy()
    df["split"] = "train"
    label_to_idx = defaultdict(list)
    
    for idx, lab in enumerate(df["label"]):
        label_to_idx[lab].append(idx)
    
    for lab, idxs in label_to_idx.items():
        idxs = np.array(idxs)
        RNG.shuffle(idxs)
        n_total = len(idxs)
        n_val = max(MIN_PER_SPLIT, int(round(n_total * VAL_RATIO)))
        n_test = max(MIN_PER_SPLIT, int(round(n_total * TEST_RATIO)))
        n_train = n_total - n_val - n_test
        if n_train < 0:
            # kam sequence bo‘lsa train minimal bo‘lsin
            n_train = max(0, n_total - n_val - n_test)
        
        df.iloc[idxs[:n_train], df.columns.get_loc("split")] = "train"
        df.iloc[idxs[n_train:n_train+n_val], df.columns.get_loc("split")] = "val"
        df.iloc[idxs[n_train+n_val:n_train+n_val+n_test], df.columns.get_loc("split")] = "test"
    
    return df

def main():
    assert os.path.exists(IN_CSV), f"Not found: {IN_CSV}"
    df = pd.read_csv(IN_CSV)
    df = stratified_split_sequence(df)

    df[df["split"]=="train"].to_csv(OUT_TRAIN, index=False)
    df[df["split"]=="val"  ].to_csv(OUT_VAL,   index=False)
    df[df["split"]=="test" ].to_csv(OUT_TEST,  index=False)

    print("✅ Saved:")
    print(" -", OUT_TRAIN, len(df[df.split=='train']))
    print(" -", OUT_VAL,   len(df[df.split=='val']))
    print(" -", OUT_TEST,  len(df[df.split=='test']))

    print("\nCounts by split & label:")
    print(df.groupby(["split","label"]).size())

if __name__ == "__main__":
    main()
