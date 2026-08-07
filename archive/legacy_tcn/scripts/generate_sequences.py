# generate_sequences.py
"""
generate_sequences.py
frame_labels_all.csv + pose_features.csv -> fall_sequences_priority_label.csv

- Label = markaziy (center) frame'ning labeli
- Class-based stride bilan no_fall oynalar siyraklashtiriladi
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ====== PATHLAR ======
PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
LABELS_CSV   = os.path.join(PROJECT_ROOT, "frame_labels_all.csv")
POSE_CSV     = os.path.join(PROJECT_ROOT, "pose_features.csv")
OUT_CSV      = os.path.join(PROJECT_ROOT, "fall_sequences_priority_label.csv")

# ====== PARAMETRLAR ======
SEQ_LEN = 30
BASE_STEP = 1
CLASS_STRIDE = {
    "no_fall": 10,
    "pre_fall": 5,
    "fall": 5,
}
# Per-clip per-class cap (ixtiyoriy). Bo'sh qoldirsangiz cheklanmaydi.
MAX_PER_CLIP_CLASS = {
    # "no_fall": 400,
    # "pre_fall": 600,
    # "fall": 600,
}

LABEL2ID = {"no_fall": 0, "pre_fall": 1, "fall": 2}

def get_nkp_cols():
    cols = []
    for j in range(17):
        cols.append(f"nkp{j}_x")
        cols.append(f"nkp{j}_y")
    return cols

NKP_COLS = get_nkp_cols()
FEAT_DIM = len(NKP_COLS)


def load_and_join():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Topilmadi: {LABELS_CSV}")
    if not os.path.exists(POSE_CSV):
        raise FileNotFoundError(f"Topilmadi: {POSE_CSV}")

    print("[INFO] Loading CSVs...")
    df_labels = pd.read_csv(LABELS_CSV)
    df_pose   = pd.read_csv(POSE_CSV)
    
    print(f"[INFO] Labels CSV: {len(df_labels)} rows")
    print(f"[INFO] Pose CSV: {len(df_pose)} rows")

    # ✅ Pose CSV da label bor yoki yo'qligini tekshirish
    if "label" in df_pose.columns and "label_id" in df_pose.columns:
        print("[INFO] Pose CSV da label mavjud, to'g'ridan-to'g'ri ishlatiladi.")
        df = df_pose.copy()
    else:
        print("[INFO] Pose CSV da label yo'q, Labels CSV dan merge qilinmoqda...")
        # Labels CSV dan kerakli ustunlarni olish
        need_lab = {"filename", "label"}
        if not need_lab.issubset(df_labels.columns):
            raise ValueError(f"Labels CSV uchun kerak: {need_lab}")
        
        # Merge qilish
        merge_cols = ["filename", "label"]
        if "label_id" in df_labels.columns:
            merge_cols.append("label_id")
        
        df = df_pose.merge(df_labels[merge_cols], on="filename", how="left")
        
        # label_id yo'q bo'lsa yaratish
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
    
    # ✅ label ustuni borligini tekshirish
    if "label" not in df.columns:
        raise ValueError("❌ Merge qilingandan keyin ham 'label' ustuni yo'q!")
    
    # ✅ NaN labellarni tekshirish
    nan_count = df["label"].isna().sum()
    if nan_count > 0:
        print(f"[WARN] {nan_count} ta frame uchun label topilmadi (NaN), ular o'chiriladi.")
        df = df.dropna(subset=["label"])
    
    # ✅ nkp ustunlarini tekshirish
    need_pose = set(NKP_COLS)
    missing_cols = need_pose - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Pose CSV da quyidagi ustunlar yo'q: {missing_cols}")

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").fillna(-1).astype(int)
    df["label_id"] = df["label_id"].astype(int)
    
    print(f"[INFO] Merged DF: {len(df)} rows")
    print("[INFO] Label distribution:")
    print(df["label"].value_counts())
    
    return df


def make_sequences_for_group(gdf, seq_len=SEQ_LEN):
    """Bitta (subject, activity, clip) guruhi uchun oynalarni hosil qiladi."""
    gdf = gdf.sort_values(by=["frame", "filename"]).reset_index(drop=True)
    T = len(gdf)
    if T < seq_len:
        return []

    feats = gdf[NKP_COLS].to_numpy(dtype=np.float32)   # [T, FEAT_DIM]
    labels = gdf["label"].astype(str).to_numpy()
    label_ids = gdf["label_id"].to_numpy(dtype=np.int64)
    frames = gdf["frame"].to_numpy()

    # clip_id (string) — per-clip-class cap uchun kerak
    clip_id = str(gdf["clip"].iloc[0]) if "clip" in gdf.columns else "clip0"

    rows = []
    clip_class_counter = defaultdict(int)

    for start in range(0, T - seq_len + 1, BASE_STEP):
        end = start + seq_len - 1
        center = start + seq_len // 2

        lab = labels[center]
        lab_id = int(label_ids[center])

        # class-based stride (fallback: BASE_STEP)
        stride = CLASS_STRIDE.get(lab, BASE_STEP)
        if (start % stride) != 0:
            continue

        key = (clip_id, lab)  # <-- MUHIM: 'key' har doim aniqlanadi

        # Per-clip per-class limit (ixtiyoriy)
        if MAX_PER_CLIP_CLASS:
            if clip_class_counter[key] >= MAX_PER_CLIP_CLASS.get(lab, 10**9):
                continue

        win = feats[start:end+1, :]                 # [seq_len, FEAT_DIM]
        flat = win.reshape(-1).astype(float)        # [seq_len * FEAT_DIM]

        row = {
            "subject": gdf["subject"].iloc[0],
            "activity": gdf["activity"].iloc[0],
            "clip": clip_id,
            "start_frame": int(frames[start]),
            "end_frame": int(frames[end]),
            "center_frame": int(frames[center]),
            "label": lab,
            "label_id": lab_id,
            "seq_len": seq_len,
            "feat_dim": FEAT_DIM,
        }
        for i, v in enumerate(flat):
            row[f"f{i}"] = v

        rows.append(row)
        clip_class_counter[key] += 1   # <-- endi xato bermaydi

    return rows


def run():
    print("[INFO] Join qilinmoqda...")
    df = load_and_join()

    print(f"[INFO] Creating sequences (seq_len={SEQ_LEN})...")
    all_rows = []
    for (subj, act, clip), g in df.groupby(["subject", "activity", "clip"], dropna=False):
        rows = make_sequences_for_group(g, seq_len=SEQ_LEN)
        if rows:
            all_rows.extend(rows)
            if len(all_rows) % 1000 == 0:
                print(f"[INFO] Progress: {len(all_rows)} sequences...")

    if not all_rows:
        raise RuntimeError("Hech qanday sequence hosil bo'lmadi. Parametrlarni tekshiring.")

    out_df = pd.DataFrame(all_rows)
    print("\n[INFO] Sequence counts by label:")
    print(out_df["label"].value_counts())

    print("[INFO] Shuffling sequences...")
    out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("✅ Saved:", OUT_CSV)
    print(f"[INFO] seq_len={SEQ_LEN}, feat_dim={FEAT_DIM}, total_sequences={len(out_df)}")
    print(f"[INFO] CSV size: {os.path.getsize(OUT_CSV) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    run()