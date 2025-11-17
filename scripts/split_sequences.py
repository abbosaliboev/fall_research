# -*- coding: utf-8 -*-
# scripts/split_sequences.py
"""
Sequences ni train/val/test ga bo'lish
- Subject-based split: bir subject faqat bitta split da bo'ladi (data leakage oldini olish)
- Stratified by label: har label uchun proporsional split
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict

PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
IN_CSV  = os.path.join(PROJECT_ROOT, "fall_sequences_priority_label.csv")
OUT_TRAIN = os.path.join(PROJECT_ROOT, "sequences_train.csv")
OUT_VAL   = os.path.join(PROJECT_ROOT, "sequences_val.csv")
OUT_TEST  = os.path.join(PROJECT_ROOT, "sequences_test.csv")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RNG = np.random.default_rng(42)

# Minimal sequence per split (har label uchun)
# Pre_fall kam bo'lgani uchun juda past qiymat
MIN_PER_SPLIT = 3  # eski: 5, yangi: 3 (pre_fall 44 ta)


def subject_based_split(df):
    """
    Subject-based split: har subject faqat bitta split da
    Bu data leakage ni to'sib qo'yadi (test da train dagi odamlar bo'lmasligi uchun)
    """
    subjects = df["subject"].unique()
    RNG.shuffle(subjects)
    
    n_total = len(subjects)
    n_train = int(round(n_total * TRAIN_RATIO))
    n_val = int(round(n_total * VAL_RATIO))
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train+n_val]
    test_subjects = subjects[n_train+n_val:]
    
    df = df.copy()
    df["split"] = "train"
    df.loc[df["subject"].isin(val_subjects), "split"] = "val"
    df.loc[df["subject"].isin(test_subjects), "split"] = "test"
    
    return df


def stratified_split_by_label(df):
    """
    Label-based stratified split (subject-agnostic)
    Har label uchun minimal val/test count saqlagan holda split
    """
    df = df.copy()
    df["split"] = "train"
    
    label_to_idx = defaultdict(list)
    for idx, lab in enumerate(df["label"]):
        label_to_idx[lab].append(idx)
    
    for lab, idxs in label_to_idx.items():
        idxs = np.array(idxs)
        RNG.shuffle(idxs)
        n_total = len(idxs)
        
        # Agar juda kam sequence bo'lsa (MIN_PER_SPLIT * 3 dan kam)
        if n_total < MIN_PER_SPLIT * 3:
            print(f"[WARN] {lab}: juda kam sequence ({n_total}), faqat train va val ga bo'linadi")
            n_train = max(1, int(n_total * 0.80))
            n_val = n_total - n_train
            n_test = 0
        else:
            # Normal split
            n_val = max(MIN_PER_SPLIT, int(round(n_total * VAL_RATIO)))
            n_test = max(MIN_PER_SPLIT, int(round(n_total * TEST_RATIO)))
            n_train = n_total - n_val - n_test
            
            # Agar train negative bo'lsa
            if n_train < MIN_PER_SPLIT:
                print(f"[WARN] {lab}: kam sequence ({n_total}), proportional split qilinadi")
                n_train = max(1, int(n_total * TRAIN_RATIO))
                n_val = max(1, int(n_total * VAL_RATIO))
                n_test = max(0, n_total - n_train - n_val)
        
        # Split assignment
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train+n_val] if n_val > 0 else []
        test_idx = idxs[n_train+n_val:n_train+n_val+n_test] if n_test > 0 else []
        
        df.loc[train_idx, "split"] = "train"
        if len(val_idx) > 0:
            df.loc[val_idx, "split"] = "val"
        if len(test_idx) > 0:
            df.loc[test_idx, "split"] = "test"
        
        print(f"  {lab:10s}: total={n_total:4d} | train={n_train:4d} | val={n_val:4d} | test={n_test:4d}")
    
    return df


def main():
    assert os.path.exists(IN_CSV), f"Not found: {IN_CSV}"
    
    print("[INFO] Loading sequences CSV...")
    df = pd.read_csv(IN_CSV)
    
    if len(df) == 0:
        raise ValueError(f"❌ CSV bo'sh: {IN_CSV}")
    
    print(f"[INFO] Total sequences: {len(df)}")
    print("[INFO] Label distribution:")
    label_counts = df["label"].value_counts()
    print(label_counts)
    
    # Subject borligini tekshirish
    has_subject = "subject" in df.columns and df["subject"].notna().any()
    
    print(f"\n[INFO] Split strategy:")
    if has_subject:
        unique_subjects = df["subject"].nunique()
        print(f"  - Found {unique_subjects} unique subjects")
        
        # Agar faqat 1 ta subject bo'lsa, subject-based split ishlamaydi
        if unique_subjects == 1:
            print(f"  ⚠️  Only 1 subject found - using LABEL-BASED split")
            has_subject = False
        else:
            print(f"  - Using SUBJECT-BASED split (recommended)")
    else:
        print(f"  - No subject column / all NaN")
        print(f"  - Using LABEL-BASED stratified split")
    
    # Split qilish
    if has_subject:
        df = subject_based_split(df)
    else:
        print(f"\n[INFO] Per-label split:")
        df = stratified_split_by_label(df)
    
    # Check splits not empty
    for split_name in ["train", "val", "test"]:
        count = (df["split"] == split_name).sum()
        if count == 0:
            print(f"⚠️  WARNING: {split_name} split is empty!")
    
    # Split qilish
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()
    
    # 'split' ustunini o'chirish (kerak emas)
    df_train = df_train.drop(columns=["split"])
    df_val = df_val.drop(columns=["split"])
    df_test = df_test.drop(columns=["split"])
    
    # Save
    df_train.to_csv(OUT_TRAIN, index=False)
    df_val.to_csv(OUT_VAL, index=False)
    df_test.to_csv(OUT_TEST, index=False)
    
    print("\n✅ Saved:")
    print(f"  - {OUT_TRAIN}: {len(df_train)} sequences")
    print(f"  - {OUT_VAL}: {len(df_val)} sequences")
    print(f"  - {OUT_TEST}: {len(df_test)} sequences")
    
    if len(df_val) == 0 or len(df_test) == 0:
        print("\n⚠️  CRITICAL WARNING: Val yoki Test bo'sh!")
        print("    MIN_PER_SPLIT ni kamaytirib qayta urining.")
    
    print("\n--- Split distribution by label ---")
    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(split_counts)
    
    print("\n--- Percentages ---")
    for split in ["train", "val", "test"]:
        pct = len(df[df.split == split]) / len(df) * 100
        print(f"{split:6s}: {pct:5.1f}%")
    
    # Subject-based split bo'lsa, subject overlap tekshirish
    if has_subject:
        train_subj = set(df_train["subject"].unique())
        val_subj = set(df_val["subject"].unique())
        test_subj = set(df_test["subject"].unique())
        
        overlap_tv = train_subj & val_subj
        overlap_tt = train_subj & test_subj
        overlap_vt = val_subj & test_subj
        
        if overlap_tv or overlap_tt or overlap_vt:
            print("\n⚠️  WARNING: Subject overlap detected!")
            print(f"  Train-Val: {overlap_tv}")
            print(f"  Train-Test: {overlap_tt}")
            print(f"  Val-Test: {overlap_vt}")
        else:
            print("\n✅ No subject overlap - clean split!")


if __name__ == "__main__":
    main()