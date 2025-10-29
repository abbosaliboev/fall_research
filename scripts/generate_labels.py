# -*- coding: utf-8 -*-
"""
generate_labels.py
Fall + No-Fall datasetlar uchun frame-level label CSV generatsiyasi.

Chiqish: <PROJECT_ROOT>/frame_labels_all.csv
Ustunlar: filename, label, label_id, subject, activity, clip, frame
"""

import os
import glob
import pandas as pd

# VS Code dagi papka nomi: FALL_RESEARCH
PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"

# Data papkasi ichida
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# fall_data/Subject1/...  
_FALL_CANDIDATES = [
    os.path.join(DATA_ROOT, "fall_data", "Subject1"),
    os.path.join(DATA_ROOT, "fall data", "Subject1"),  # eski nom variant
]
FALL_BASE = next((p for p in _FALL_CANDIDATES if os.path.exists(p)), _FALL_CANDIDATES[0])

# nofall_data/Subject1/...
NOFALL_BASE = os.path.join(DATA_ROOT, "nofall_data")  # ichida Subject1 bor

IMG_EXTS = (".png", ".jpg", ".jpeg")
OUT_CSV = os.path.join(PROJECT_ROOT, "frame_labels_all.csv")
LABEL2ID = {"no_fall": 0, "pre_fall": 1, "fall": 2}

# ============ FALL ANNOTATSIYALARI (siz bergan) ============
annotations = {
    "Activity1": {
        1: {'no_fall': (1, 70), 'pre_fall': (71, 80), 'fall': (81, 112)},
        2: {'no_fall': (1, 70), 'pre_fall': (71, 80), 'fall': (81, 112)},
        3: {'no_fall': (1, 20), 'pre_fall': (21, 35), 'fall': (36, 64)},
        4: {'no_fall': (1, 20), 'pre_fall': (21, 35), 'fall': (36, 64)},
        5: {'no_fall': (1, 20), 'pre_fall': (21, 36), 'fall': (37, 64)},
        6: {'no_fall': (1, 20), 'pre_fall': (21, 36), 'fall': (37, 64)},
    },
    "Activity2": {
        1: {'no_fall': (1, 15), 'pre_fall': (16, 35), 'fall': (36, 64)},
        2: {'no_fall': (1, 15), 'pre_fall': (16, 35), 'fall': (36, 64)},
        3: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
        4: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
        5: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
        6: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
    },
    "Activity3": {
        1: {'no_fall': (1, 24), 'pre_fall': (25, 40), 'fall': (41, 70)},
        2: {'no_fall': (1, 24), 'pre_fall': (25, 40), 'fall': (41, 70)},
        3: {'no_fall': (1, 24), 'pre_fall': (25, 40), 'fall': (41, 70)},
        4: {'no_fall': (1, 24), 'pre_fall': (25, 40), 'fall': (41, 70)},
        5: {'no_fall': (1, 20), 'pre_fall': (21, 35), 'fall': (36, 64)},
        6: {'no_fall': (1, 20), 'pre_fall': (21, 35), 'fall': (36, 64)},
    },
    "Activity4": {
        1: {'no_fall': (1, 15), 'pre_fall': (16, 25), 'fall': (26, 40)},
        2: {'no_fall': (1, 15), 'pre_fall': (16, 25), 'fall': (26, 40)},
        3: {'no_fall': (1, 20), 'pre_fall': (21, 30), 'fall': (31, 48)},
        4: {'no_fall': (1, 20), 'pre_fall': (21, 30), 'fall': (31, 48)},
        5: {'no_fall': (1, 24), 'pre_fall': (25, 33), 'fall': (34, 48)},
        6: {'no_fall': (1, 24), 'pre_fall': (25, 33), 'fall': (34, 48)},
    },
    "Activity5": {
        1: {'no_fall': (1, 15), 'pre_fall': (16, 36), 'fall': (37, 64)},
        2: {'no_fall': (1, 15), 'pre_fall': (16, 36), 'fall': (37, 64)},
        3: {'no_fall': (1, 20), 'pre_fall': (21, 36), 'fall': (37, 64)},
        4: {'no_fall': (1, 20), 'pre_fall': (21, 36), 'fall': (37, 64)},
        5: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
        6: {'no_fall': (1, 20), 'pre_fall': (21, 40), 'fall': (41, 64)},
    },
}

def _try_extensions(base_no_ext: str):
    for ext in IMG_EXTS:
        cand = base_no_ext + ext
        if os.path.exists(cand):
            return cand
    return None

def add_fall(rows: list):
    missing = 0
    for activity, folders in annotations.items():
        for folder_id, classes in folders.items():
            for label, (start, end) in classes.items():
                for i in range(start, end + 1):
                    base_no_ext = os.path.join(FALL_BASE, activity, str(folder_id), f"img_{i:06d}")
                    abs_path = _try_extensions(base_no_ext)
                    if abs_path is None:
                        missing += 1
                        continue
                    rel = os.path.relpath(abs_path, start=PROJECT_ROOT)
                    rows.append(dict(
                        filename=rel,
                        label=label,
                        label_id=LABEL2ID[label],
                        subject="Subject1",
                        activity=activity,
                        clip=str(folder_id),
                        frame=i
                    ))
    if missing:
        print(f"[WARN] FALL: topilmagan rasm/frame: {missing} ta")

def add_nofall(rows: list):
    if not os.path.exists(NOFALL_BASE):
        print(f"[INFO] NOFALL_BASE yo‘q: {NOFALL_BASE}")
        return

    # 1-qavat: Subject (Subject1, ...)
    level1 = sorted(glob.glob(os.path.join(NOFALL_BASE, "*"))) or [NOFALL_BASE]
    total = 0

    for l1 in level1:
        subj = os.path.basename(l1) if l1 != NOFALL_BASE else "NoFallSet"

        # 2-qavat: Activity (Bend, Sit, Walk, Jump, ...)
        level2 = sorted(glob.glob(os.path.join(l1, "*"))) or [l1]
        for l2 in level2:
            act = os.path.basename(l2) if l2 != l1 else "NoFall"

            # Case A: rasmlar bevosita shu papkada
            imgs = sorted(
                glob.glob(os.path.join(l2, "*.png")) +
                glob.glob(os.path.join(l2, "*.jpg")) +
                glob.glob(os.path.join(l2, "*.jpeg"))
            )
            if imgs:
                clip = act
                for img in imgs:
                    base = os.path.splitext(os.path.basename(img))[0]
                    try: frame_idx = int(base)
                    except: frame_idx = -1
                    rel = os.path.relpath(img, start=PROJECT_ROOT)
                    rows.append(dict(
                        filename=rel,
                        label="no_fall",
                        label_id=LABEL2ID["no_fall"],
                        subject=subj,
                        activity=act,
                        clip=clip,
                        frame=frame_idx
                    ))
                    total += 1
                continue

            # Case B: ichki papkalarda rasmlar (cliplar)
            clips = [d for d in sorted(glob.glob(os.path.join(l2, "*"))) if os.path.isdir(d)]
            for c in clips:
                clip = os.path.basename(c)
                imgs = sorted(
                    glob.glob(os.path.join(c, "*.png")) +
                    glob.glob(os.path.join(c, "*.jpg")) +
                    glob.glob(os.path.join(c, "*.jpeg"))
                )
                for img in imgs:
                    base = os.path.splitext(os.path.basename(img))[0]
                    try: frame_idx = int(base)
                    except: frame_idx = -1
                    rel = os.path.relpath(img, start=PROJECT_ROOT)
                    rows.append(dict(
                        filename=rel,
                        label="no_fall",
                        label_id=LABEL2ID["no_fall"],
                        subject=subj,
                        activity=act,
                        clip=clip,
                        frame=frame_idx
                    ))
                    total += 1

    print(f"[INFO] NO-FALL qo‘shildi: {total} kadr")

def main():
    print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    print("[INFO] FALL_BASE    :", FALL_BASE)
    print("[INFO] NOFALL_BASE  :", NOFALL_BASE)

    rows = []
    if os.path.exists(FALL_BASE):
        add_fall(rows)
    else:
        print(f"[WARN] FALL_BASE topilmadi: {FALL_BASE}")

    add_nofall(rows)

    if not rows:
        print("[ERR] Hech narsa topilmadi. Yo‘llarni tekshiring.")
        return

    df = pd.DataFrame(rows, columns=["filename","label","label_id","subject","activity","clip","frame"])
    before = len(df)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    if len(df) < before:
        print(f"[INFO] Dublikatsiya olib tashlandi: {before - len(df)} satr")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("✅ Saved:", OUT_CSV)
    print("\n--- Label counts ---")
    print(df["label"].value_counts())
    print("\n--- Misol satrlar ---")
    print(df.head(8))

if __name__ == "__main__":
    main()
