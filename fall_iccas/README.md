# Fall Detection — ST-GCN + Physics Rescue
**ICCAS 2026 paper project**

Kamera tasviridan yiqilishni aniqlash: ST-GCN skeleton modeli + fizika qoidalari filtri (2-stage).

---

## Arxitektura

```
Kamera kadrlari
      │
      ▼
YOLO11n-pose  →  17 COCO keypointlar (x, y, conf)
      │
      ▼
Sliding window (T=30, stride=15, ~19 FPS)
      │
      ▼
ST-GCN (Stage 1)  →  fall probability p
      │
      ├── p >= 0.55  →  FALL  (ishonchli)
      ├── 0.50 <= p < 0.55  →  Physics Filter qaror beradi (Rescue zone)
      └── p < 0.50  →  NO-FALL
```

**Physics Filter** (Stage 2 — Rescue mode):
- Hip Y koordinatiga Butterworth low-pass filter
- Velocity va acceleration threshold larini hisoblaydi
- Faqat Stage 1 noaniq bo'lganda ishga kiradi (hech qachon Stage 1 topganlarni o'chirmaydi)

---

## Dataset

**UP-Fall Detection Dataset** (Martínez-Villaseñor et al., Sensors 2019)
- 17 subject, 11 activity, 3 trial
- Camera1 rasmlari ishlatildi
- Activities 1–5 = FALL, Activities 6–11 = NO-FALL

| Activity | Nomi | Label |
|---|---|---|
| 1 | Falling forward (hands) | FALL |
| 2 | Falling forward (knees) | FALL |
| 3 | Falling sideways | FALL |
| 4 | Falling backward | FALL |
| 5 | Hitting obstacle | FALL |
| 6 | Sitting abruptly | NO-FALL |
| 7 | Walking | NO-FALL |
| 8 | Standing | NO-FALL |
| 9 | Sitting | NO-FALL |
| 10 | Picking up object | NO-FALL |
| 11 | Jumping | NO-FALL |

---

## Fayllar

```
fall_iccas/
├── dataset/                    # UP-Fall rasmlari (Subject1-17)
├── cv_dataset/
│   ├── X.npy                   # (N, 30, 17, 3) — keypoint sequences
│   ├── y.npy                   # (N,) — 0/1 labels
│   └── meta.csv                # subject/activity/trial info
├── checkpoints/
│   ├── best_stgcn.pth          # eng yaxshi ST-GCN model
│   └── two_stage_config.json   # threshold lar
├── stgcn/
│   ├── model.py                # ST-GCN arxitekturasi
│   ├── graph.py                # 17-joint COCO skeleton grafik
│   ├── physics.py              # PhysicsFilter
│   └── two_stage.py            # TwoStageDetector (Rescue mode)
├── prepare_cv_dataset.py       # YOLO keypoint extraction
├── train_two_stage.py          # To'liq training pipeline
└── label_dataset.py            # Sensor CSV labeling (ixtiyoriy)
```

---

## Ishlatish

### 1. Dataset tayyorlash
```bash
python prepare_cv_dataset.py
```
`cv_dataset/X.npy`, `y.npy`, `meta.csv` hosil bo'ladi.

### 2. Train
```bash
python train_two_stage.py
```
Natijalar terminalga chiqadi, model `checkpoints/` ga saqlanadi.

---

## Hozirgi natijalar (Subject 1 only — subject-dependent)

| Model | Accuracy | Fall F1 | Precision | Recall |
|---|---|---|---|---|
| ST-GCN (Stage 1) | 98.8% | 0.960 | 0.96 | 0.96 |
| ST-GCN + Physics Rescue | 98.8% | 0.960 | 0.96 | 0.96 |

> **Muhim:** Bu natijalar Subject 1 da train/test qilingan (subject-dependent).
> Haqiqiy baholash uchun LOSO (Leave-One-Subject-Out) kerak — Subject 2–17 yuklanganidan keyin.

---

## Muhim texnik sozlamalar

| Parametr | Qiymat |
|---|---|
| YOLO model | yolo11n-pose.pt |
| YOLO conf threshold | 0.1 (past, yiqilish posalarini aniqlash uchun) |
| Window size | 30 frame |
| Stride | 15 frame |
| FPS | ~19 Hz |
| ST-GCN channels | 64→64→64→128→128→128→256→256→256 |
| Epochs | 60 |
| Optimizer | Adam lr=1e-3, CosineAnnealingLR |
| GPU | NVIDIA TITAN RTX 24GB |

---

## Hal qilingan muammolar

1. **Zero frame muammosi** — Yiqilish paytida YOLO odamni topa olmay 0 qo'ygan (14.5% fall frame). Yechim: `conf=0.1` + forward-fill interpolation → 0% zero frame.
2. **Camera papka nomi xatosi** — Barcha trialda `Activity2` nomi turgan. PowerShell script bilan to'g'rilandi (54 ta papka).
3. **Physics filter zararli bo'lishi** — Eski AND mantiq Stage 1 topganlarni o'chirgan. Yangi "Rescue" mantiq bilan hal qilindi (physics faqat noaniq holatlarda ishlaydi).
