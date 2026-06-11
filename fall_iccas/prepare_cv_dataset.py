"""
CV dataset preparation for TCN / ST-GCN fall detection.
Keypoint extractor: YOLOv11n-pose  (17 COCO joints)

Pipeline:
  images -> YOLO-pose -> 17 keypoints (x, y, conf) per frame
         -> sliding-window sequences
         -> X.npy  (N, T, V, C)   N sequences, T=30 frames, V=17 joints, C=3 (x,y,conf)
         -> y.npy  (N,)            0=no-fall  1=fall
         -> meta.csv

Normalized coords: x,y divided by image width/height -> [0,1]
Confidence from YOLO kept as 3rd channel.

ST-GCN: X.transpose(0,3,1,2)[:,None] -> (N, C, T, V, 1)
TCN   : X.reshape(N, T, -1)          -> (N, T, V*C)
"""

import os
import re
import csv
import numpy as np
import cv2
from ultralytics import YOLO

# ── config ────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "cv_dataset")
CAMERA      = "Camera1"
WINDOW_SIZE = 30
STRIDE      = 15
FALL_ACTIVITIES = {1, 2, 3, 4, 5}
N_JOINTS    = 17
N_COORDS    = 3     # x, y, confidence

os.makedirs(OUT_DIR, exist_ok=True)

# load once
MODEL = YOLO("yolo11n-pose.pt")   # downloads automatically if not present


def sorted_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in exts)


def extract_keypoints_batch(img_paths, batch_size=8):
    """
    Run YOLO-pose on a list of images in batches.
    Returns (F, 17, 3) array — x, y normalized, confidence.
    Zero-filled if no person detected.
    """
    import gc
    results_all = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i: i + batch_size]
        results = MODEL(batch, verbose=False, conf=0.1)
        results_all.extend(results)
        gc.collect()

    kps = np.zeros((len(img_paths), N_JOINTS, N_COORDS), dtype=np.float32)
    for i, res in enumerate(results_all):
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            continue
        # take highest-confidence person
        if res.keypoints.conf is not None:
            person_idx = int(res.keypoints.conf.sum(dim=1).argmax())
        else:
            person_idx = 0

        xy   = res.keypoints.xy[person_idx].cpu().numpy()    # (17, 2) pixels
        conf = res.keypoints.conf[person_idx].cpu().numpy()  # (17,)

        # read image size for normalization
        h, w = res.orig_shape
        xy_norm = xy / np.array([w, h], dtype=np.float32)

        kps[i, :, 0] = xy_norm[:, 0]   # x normalized
        kps[i, :, 1] = xy_norm[:, 1]   # y normalized
        kps[i, :, 2] = conf             # confidence

    return kps   # (F, 17, 3)


def interpolate_zero_frames(kps: np.ndarray) -> np.ndarray:
    """
    Fill zero frames (failed detections) using forward-fill then backward-fill.
    A frame is zero when all xy coordinates sum to 0.
    kps : (F, 17, 3)
    """
    kps = kps.copy()
    F = len(kps)
    is_zero = kps[:, :, :2].sum(axis=(1, 2)) == 0  # (F,)

    if not is_zero.any():
        return kps

    # forward fill
    last_valid = None
    for i in range(F):
        if not is_zero[i]:
            last_valid = kps[i].copy()
        elif last_valid is not None:
            kps[i] = last_valid

    # backward fill for leading zeros
    first_valid = None
    for i in range(F):
        if kps[i, :, :2].sum() > 0:
            first_valid = kps[i].copy()
            break
    if first_valid is not None:
        for i in range(F):
            if kps[i, :, :2].sum() == 0:
                kps[i] = first_valid
            else:
                break

    return kps


def get_trials(root):
    for subj_name in sorted(os.listdir(root)):
        sm = re.match(r"Subject(\d+)$", subj_name)
        if not sm:
            continue
        subj_id = int(sm.group(1))
        subj_path = os.path.join(root, subj_name)

        for act_name in sorted(os.listdir(subj_path)):
            am = re.match(r"Activity(\d+)$", act_name)
            if not am:
                continue
            act_id = int(am.group(1))
            act_path = os.path.join(subj_path, act_name)

            for trial_name in sorted(os.listdir(act_path)):
                tm = re.match(r"Trial(\d+)$", trial_name)
                if not tm:
                    continue
                trial_id = int(tm.group(1))
                trial_path = os.path.join(act_path, trial_name)

                cam_folder = None
                for d in os.listdir(trial_path):
                    if CAMERA in d and os.path.isdir(os.path.join(trial_path, d)):
                        cam_folder = os.path.join(trial_path, d)
                        break
                if cam_folder is None:
                    continue

                imgs = sorted_images(cam_folder)
                if not imgs:
                    continue

                yield subj_id, act_id, trial_id, cam_folder, imgs


def main():
    all_X, all_y, meta_rows = [], [], []
    seq_id = 0

    for subj_id, act_id, trial_id, cam_folder, imgs in get_trials(DATASET_DIR):
        label = 1 if act_id in FALL_ACTIVITIES else 0
        print(f"  S{subj_id} A{act_id} T{trial_id}  {len(imgs)} frames  label={label}")

        img_paths = [os.path.join(cam_folder, f) for f in imgs]
        trial_kps = extract_keypoints_batch(img_paths)  # (F, 17, 3)

        # fill failed detections from neighboring frames
        zeros_before = int((trial_kps[:, :, :2].sum(axis=(1, 2)) == 0).sum())
        trial_kps = interpolate_zero_frames(trial_kps)
        zeros_after = int((trial_kps[:, :, :2].sum(axis=(1, 2)) == 0).sum())
        if zeros_before > 0:
            print(f"    zero frames: {zeros_before} -> {zeros_after} after interpolation")

        F = len(trial_kps)
        if F < WINDOW_SIZE:
            pad = np.tile(trial_kps[-1:], (WINDOW_SIZE - F, 1, 1))
            trial_kps = np.concatenate([trial_kps, pad], axis=0)
            F = WINDOW_SIZE

        start = 0
        while start + WINDOW_SIZE <= F:
            window = trial_kps[start: start + WINDOW_SIZE]  # (T, V, C)
            all_X.append(window)
            all_y.append(label)
            meta_rows.append({
                "seq_id": seq_id, "subject": subj_id,
                "activity": act_id, "trial": trial_id,
                "start_frame": start, "label": label,
            })
            seq_id += 1
            start += STRIDE

    X = np.array(all_X, dtype=np.float32)  # (N, T, V, C)
    y = np.array(all_y, dtype=np.int64)

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)

    with open(os.path.join(OUT_DIR, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_id","subject","activity","trial","start_frame","label"])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"\nX shape  : {X.shape}   (N, T={WINDOW_SIZE}, V={N_JOINTS}, C={N_COORDS})")
    print(f"y shape  : {y.shape}")
    print(f"FALL     : {(y==1).sum()} sequences")
    print(f"NO-FALL  : {(y==0).sum()} sequences")
    print(f"\nSaved to : {OUT_DIR}")
    print("\nST-GCN input: X.transpose(0,3,1,2)[:,None]  -> (N, C, T, V, 1)")
    print("TCN input   : X.reshape(N, T, -1)           -> (N, T, V*C)")


if __name__ == "__main__":
    main()
