# Model Arxitekturasi

## Umumiy ko'rinish

```
Input: kamera kadrlari (Camera1, ~19 FPS)
         │
         ▼
  ┌─────────────────┐
  │  YOLO11n-pose   │  conf=0.1, batch_size=8
  │  (keypoint det) │
  └────────┬────────┘
           │  (F, 17, 3) — frame, joint, [x, y, conf]
           ▼
  ┌─────────────────┐
  │ Zero-frame fill │  forward-fill → backward-fill
  │ (interpolation) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Sliding window  │  T=30, stride=15
  └────────┬────────┘
           │  (N, 30, 17, 3)
           ▼
  ┌─────────────────┐
  │   ST-GCN        │  Stage 1
  │   (9 blocks)    │
  └────────┬────────┘
           │  fall probability p
           ▼
  ┌───────────────────────────────────┐
  │        Physics Rescue             │  Stage 2
  │  p >= 0.55  → FALL               │
  │  0.50 <= p < 0.55 → physics?     │
  │  p < 0.50   → NO-FALL            │
  └───────────────────────────────────┘
```

---

## ST-GCN (Spatial Temporal Graph Convolutional Network)

**Asosiy manba:** Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018.

### Skeleton grafik

17 ta COCO keypoint (YOLO11n-pose chiqishi):

```
0: nose          5: left_shoulder    11: left_hip
1: left_eye      6: right_shoulder   12: right_hip
2: right_eye     7: left_elbow       13: left_knee
3: left_ear      8: right_elbow      14: right_knee
4: right_ear     9: left_wrist       15: left_ankle
                10: right_wrist      16: right_ankle
```

**Adjacency matrix A** — shape (3, 17, 17), 3 ta subset:
- `A[0]` — self-link (har joint o'zi bilan)
- `A[1]` — centripetal (yelkalar, sonlar → markazga)
- `A[2]` — centrifugal (markazdan → qo'l, oyoq uchlari)

Center node: `11` (left_hip, BFS root)

### Arxitektura

```
Input: (N, C=3, T=30, V=17, M=1)

Block 1-3:   SpatialGCN(3→64)  + TemporalConv(64, stride=1)
Block 4-6:   SpatialGCN(64→128) + TemporalConv(128, stride=2 block4)
Block 7-9:   SpatialGCN(128→256) + TemporalConv(256, stride=2 block7)

GlobalAvgPool → Dropout(0.5) → Linear(256→2)

Output: (N, 2) logits  →  softmax  →  fall probability
```

Har bir STGCNBlock:
```
SpatialGCN:
  x → einsum(A, x) → BatchNorm → ReLU
  + learnable attention mask on A

TemporalConv:
  Conv2d(C, C, kernel=(9,1), padding=(4,0)) → BatchNorm → ReLU
  + residual connection (1x1 conv if channels change)
```

### Training sozlamalari

| Parametr | Qiymat |
|---|---|
| Epochs | 60 |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=60) |
| Dropout | 0.5 |
| Class imbalance | WeightedRandomSampler (FALL:NO-FALL = 1:~6) |
| Augmentation | Horizontal flip (p=0.5) + Gaussian noise (σ=0.01) |
| Best model | val fall F1 asosida saqlanadi |

---

## Physics Filter (Stage 2)

### Hisoblash

```python
# 1. Mid-hip Y koordinatini olish (COCO index 11, 12)
hip_y = (seq[:, 11, 1] + seq[:, 12, 1]) / 2.0   # 0=top, 1=bottom

# 2. Position filter (4 Hz Butterworth lowpass)
hip_y_filtered = lowpass(hip_y, fc=4.0)

# 3. Velocity (downward = positive)
velocity = gradient(hip_y_filtered, dt=1/fps)
velocity_f = lowpass(velocity, fc=8.0)

# 4. Acceleration
acceleration = gradient(velocity_f, dt=1/fps)
acceleration_f = lowpass(acceleration, fc=6.0)

# 5. Features
max_velocity = velocity_f.max()
max_abs_acc  = abs(acceleration_f).max()
hip_drop     = hip_y_filtered.max() - hip_y_filtered.min()
```

### Thresholdlar

Validation setida grid-search orqali topiladi:
- `vel_threshold = 0.0354` (normalized units/s)
- `acc_threshold = 0.3545` (normalized units/s²)

Qaror: `max_velocity > vel_threshold AND max_abs_acc > acc_threshold` → physics confirms fall

### Rescue mantiq

```
Stage 1 prob:
  ≥ 0.55  → FALL   (Stage 1 ishonchli, physics tekmaydi)
  [0.50, 0.55)  → physics qaror beradi  ← Rescue zone
  < 0.50  → NO-FALL
```

**Muhim farq eski AND mantiqdan:**
- **Eski (AND):** `Stage1=1 AND physics=1` — physics Stage 1 topganlarni o'chirishi mumkin
- **Yangi (Rescue):** physics faqat Stage 1 MISS qilganlarni qutqarishi mumkin

---

## Dataset tayyorlash

### YOLO keypoint extraction

```python
MODEL = YOLO("yolo11n-pose.pt")
# conf=0.1 — yiqilish posalari uchun past threshold
results = MODEL(batch, conf=0.1)

# eng ishonchli odamni olish
person_idx = keypoints.conf.sum(dim=1).argmax()

# normalizatsiya
xy_norm = xy_pixels / [image_width, image_height]  # [0, 1] oralig'ida
```

### Zero-frame interpolation

```python
def interpolate_zero_frames(kps):
    # Forward fill: oxirgi aniqlanganidan foydalanish
    for i in range(len(kps)):
        if frame_is_zero(kps[i]):
            kps[i] = last_valid_frame
    # Backward fill: boshidagi zero larni to'ldirish
    ...
```

**Natija:** Fall sequencelarda zero frame 14.5% → **0%**

### Sliding window

```
Trial (F frames) → windows:
  [0:30], [15:45], [30:60], ...

Window size T=30 (~1.58 sekund at 19 FPS)
Stride     S=15 (~0.79 sekund)
```

---

## Fayl manbalar

| Fayl | Tavsif |
|---|---|
| `stgcn/graph.py` | 17-joint COCO skeleton, adjacency matrix A(3,17,17) |
| `stgcn/model.py` | ST-GCN 9 block, learnable attention |
| `stgcn/physics.py` | Butterworth filter, threshold fitting, grid-search |
| `stgcn/two_stage.py` | TwoStageDetector, Rescue mantiq, tune_thresholds() |
| `prepare_cv_dataset.py` | YOLO extraction + interpolation + windowing |
| `train_two_stage.py` | To'liq pipeline: split → train → physics fit → evaluate |
