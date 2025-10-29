# -*- coding: utf-8 -*-
# scripts/split_sequences.py
import os, pandas as pd, numpy as np
from collections import Counter

PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
IN_CSV  = os.path.join(PROJECT_ROOT, "fall_sequences_priority_label.csv")
OUT_TRAIN = os.path.join(PROJECT_ROOT, "sequences_train.csv")
OUT_VAL   = os.path.join(PROJECT_ROOT, "sequences_val.csv")
OUT_TEST  = os.path.join(PROJECT_ROOT, "sequences_test.csv")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RNG = np.random.default_rng(42)

def stratified_split_clip(df):
    # clip id
    df["clip_id"] = df["subject"].astype(str)+"|"+df["activity"].astype(str)+"|"+df["clip"].astype(str)
    # har clip uchun mode label
    modes = df.groupby("clip_id")["label"].agg(lambda s: Counter(s).most_common(1)[0][0]).reset_index()
    # label bo‘yicha cliplar ro‘yxati
    label_to_clips = {}
    for lab, g in modes.groupby("label"):
        label_to_clips[lab] = g["clip_id"].tolist()
        RNG.shuffle(label_to_clips[lab])

    train_clips, val_clips, test_clips = set(), set(), set()
    for lab, clips in label_to_clips.items():
        n = len(clips)
        n_tr = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # qolgan test
        n_te = max(0, n - n_tr - n_val)
        train_clips.update(clips[:n_tr])
        val_clips.update(clips[n_tr:n_tr+n_val])
        test_clips.update(clips[n_tr+n_val:n_tr+n_val+n_te])

    df["split"] = "train"
    df.loc[df["clip_id"].isin(val_clips), "split"] = "val"
    df.loc[df["clip_id"].isin(test_clips), "split"] = "test"
    return df

def main():
    assert os.path.exists(IN_CSV), f"Not found: {IN_CSV}"
    df = pd.read_csv(IN_CSV)
    df = stratified_split_clip(df)

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
