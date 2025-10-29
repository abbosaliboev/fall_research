# -*- coding: utf-8 -*-
"""
extract_pose.py
- frame_labels_all.csv ni o'qiydi (filename = PROJECT_ROOT ga nisbiy yo'l)
- Har rasmga YOLO-Pose ishlatib 17 keypoint: x,y,conf (raw) va normalized (nx,ny) ni yozadi
- Bir nechta odam aniqlansa, keypoint konfidentsiyalari yig'indisi eng katta bo'lganini oladi
- Aniqlanmasa: 0 bilan to'ldiriladi, detected=0

Chiqish:
  <PROJECT_ROOT>/pose_features.csv
  Ustunlar:
    filename, subject, activity, clip, frame, detected,
    kp0_x, kp0_y, kp0_c, ..., kp16_x, kp16_y, kp16_c,
    nkp0_x, nkp0_y, ..., nkp16_x, nkp16_y
"""

import os
import math
import pandas as pd
from tqdm import tqdm

# Ultralytics YOLO Pose
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics o'rnatilmagan. `pip install ultralytics opencv-python`") from e

# ============ PATHLAR ============
PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
LABELS_CSV   = os.path.join(PROJECT_ROOT, "frame_labels_all.csv")
# Weights fayllaringiz scripts/ ichida bor (screenshotdan)
WEIGHTS_CAND = [
    os.path.join(PROJECT_ROOT, "scripts", "yolo11m-pose.pt"),
    os.path.join(PROJECT_ROOT, "scripts", "yolo11s-pose.pt"),
    os.path.join(PROJECT_ROOT, "scripts", "yolov8s-pose.pt"),
    os.path.join(PROJECT_ROOT, "scripts", "yolov8n-pose.pt"),
]
OUT_CSV      = os.path.join(PROJECT_ROOT, "pose_features.csv")

# Inference sozlamalari
DEVICE = 0  # 0 -> CUDA:0, CPU bo'lsa 'cpu' yozing: DEVICE = 'cpu'
IMGSZ  = 640
CONF   = 0.25

NUM_KP = 17  # COCO format

# ------------ KP indekslari uchun qulay nomlar (COCO17) -------------
# (0:nose, 5:L-shoulder, 6:R-shoulder, 11:L-hip, 12:R-hip)
KP_L_SHO = 5
KP_R_SHO = 6
KP_L_HIP = 11
KP_R_HIP = 12


def _pick_weights():
    for p in WEIGHTS_CAND:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "YOLO-Pose weights topilmadi. Quyidagilardan biri kerak:\n" +
        "\n".join(WEIGHTS_CAND)
    )


def _person_selector(kpts_xy, kpts_conf):
    """
    Bir nechta odam bo'lsa, konfidentsiya yig'indisi eng katta bo'lganini tanlaymiz.
    kpts_xy:  (N, 17, 2)
    kpts_conf:(N, 17)
    return idx (int)
    """
    import numpy as np
    scores = kpts_conf.sum(axis=1)  # [N]
    return int(np.argmax(scores))


def _root_and_scale(kp_xy, kp_conf):
    """
    Root: mid-hip (11,12) mavjud bo'lsa; bo'lmasa mid-shoulder (5,6); undan keyin fallback -> 0,0
    Scale: shoulder distance (5-6) mavjud bo'lsa; bo'lmasa hip distance (11-12); bo'lmasa 1.0
    kp_xy: (17,2)  kp_conf: (17,)
    """
    import numpy as np

    def _ok(i): return (i is not None) and (i >= 0) and (i < NUM_KP) and (kp_conf[i] > 0.05)

    # root: mid-hip -> mid-shoulder -> (0,0)
    if _ok(KP_L_HIP) and _ok(KP_R_HIP):
        root = (kp_xy[KP_L_HIP] + kp_xy[KP_R_HIP]) / 2.0
    elif _ok(KP_L_SHO) and _ok(KP_R_SHO):
        root = (kp_xy[KP_L_SHO] + kp_xy[KP_R_SHO]) / 2.0
    else:
        root = np.array([0.0, 0.0], dtype=float)

    # scale: shoulder dist -> hip dist -> 1.0
    def _dist(a, b):
        return float(np.linalg.norm(kp_xy[a] - kp_xy[b]))
    scale = 1.0
    if _ok(KP_L_SHO) and _ok(KP_R_SHO):
        d = _dist(KP_L_SHO, KP_R_SHO)
        if d > 1e-3: scale = d
    elif _ok(KP_L_HIP) and _ok(KP_R_HIP):
        d = _dist(KP_L_HIP, KP_R_HIP)
        if d > 1e-3: scale = d

    return root, scale


def _normalize(kp_xy, kp_conf):
    """
    (x,y) -> root-center + scale-norm (see _root_and_scale)
    return nkp_xy (17,2)
    """
    import numpy as np
    root, scale = _root_and_scale(kp_xy, kp_conf)
    if not (scale > 1e-6):
        scale = 1.0
    nkp = (kp_xy - root) / scale
    return nkp


def run():
    import numpy as np

    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Labels CSV topilmadi: {LABELS_CSV}")

    weights = _pick_weights()
    print("[INFO] Weights:", weights)

    print("[INFO] YOLO Pose model yuklanmoqda...")
    model = YOLO(weights)

    df = pd.read_csv(LABELS_CSV)
    if "filename" not in df.columns:
        raise ValueError("CSVda 'filename' ustuni yo‘q")

    # Keraksiz takrorlarni olib tashlaymiz (bitta rasm bir marta infer qilinsin)
    files = df["filename"].drop_duplicates().tolist()

    rows = []
    miss = 0
    found = 0

    for relpath in tqdm(files, desc="Pose extracting"):
        abspath = os.path.join(PROJECT_ROOT, relpath)
        if not os.path.exists(abspath):
            miss += 1
            # yo'q bo'lsa skiplaymiz, lekin satr yozmaymiz
            continue

        # Inference
        try:
            results = model(
                abspath, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False
            )
        except Exception as e:
            print(f"[ERR] Inference xatosi: {abspath}\n{e}")
            miss += 1
            continue

        res = results[0]
        # kpts
        if (res.keypoints is None) or (res.keypoints.xy is None) or (len(res.keypoints.xy) == 0):
            # Hech kim topilmadi
            kp_xy = np.zeros((NUM_KP, 2), dtype=float)
            kp_c  = np.zeros((NUM_KP,), dtype=float)
            nkp_xy = np.zeros_like(kp_xy)
            detected = 0
        else:
            kxy = res.keypoints.xy.cpu().numpy()       # (N,17,2)
            kcf = res.keypoints.conf.cpu().numpy()     # (N,17)
            idx = _person_selector(kxy, kcf)
            kp_xy = kxy[idx]        # (17,2)
            kp_c  = kcf[idx]        # (17,)
            nkp_xy = _normalize(kp_xy, kp_c)
            detected = 1

        # subject/activity/clip/frame ni filename'dan olish (agar bor bo'lsa)
        # relpath misol: fall_data\Subject1\Activity1\1\23.png yoki nofall_data\Subject1\Walk\clip01\0001.png
        parts = relpath.replace("\\", "/").split("/")
        subject = ""
        activity = ""
        clip = ""
        frame = -1
        # heuristika:
        try:
            # frame
            bn = os.path.splitext(parts[-1])[0]
            frame = int(bn)
        except:
            frame = -1

   
        # fall_data bo'lsa (oldinda 'data/' bo'lishi mumkin)
        if len(parts) >= 6 and parts[1] == "fall_data":
            subject = parts[2]
            activity = parts[3]
            clip = parts[4]

        # nofall_data bo'lsa
        elif len(parts) >= 5 and parts[1] == "nofall_data":
            subject = parts[2]
            activity = parts[3]
            clip = parts[4] if len(parts) >= 6 else activity


        # satr yig'ish
        row = {
            "filename": relpath,
            "subject": subject,
            "activity": activity,
            "clip": clip,
            "frame": frame,
            "detected": detected,
        }

        # raw kps
        for j in range(NUM_KP):
            row[f"kp{j}_x"] = float(kp_xy[j, 0])
            row[f"kp{j}_y"] = float(kp_xy[j, 1])
            row[f"kp{j}_c"] = float(kp_c[j])

        # normalized kps
        for j in range(NUM_KP):
            row[f"nkp{j}_x"] = float(nkp_xy[j, 0])
            row[f"nkp{j}_y"] = float(nkp_xy[j, 1])

        rows.append(row)
        found += 1

    pose_df = pd.DataFrame(rows)
    pose_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("✅ Saved:", OUT_CSV)
    print(f"[INFO] Jami: {len(files)}, topildi: {found}, topilmadi/skip: {miss}")
    if "detected" in pose_df.columns:
        print(pose_df["detected"].value_counts(dropna=False))


if __name__ == "__main__":
    run()
