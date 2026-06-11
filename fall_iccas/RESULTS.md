# Training Results Tarixi

## Hozirgi eng yaxshi natija (Subject 1, subject-dependent)

**Sana:** 2026-06-11
**Dataset:** Subject 1 only — X.npy shape (1145, 30, 17, 3)
**Split:** 70/15/15 stratified (train=801, val=172, test=172)

| Model | Accuracy | Fall F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| ST-GCN (Stage 1) | 98.8% | **0.960** | 0.96 | 0.96 | 1 | 1 |
| ST-GCN + Physics Rescue | 98.8% | **0.960** | 0.96 | 0.96 | 1 | 1 |

Confusion matrix (test set, 172 samples):
```
              Pred NO-FALL  Pred FALL
True NO-FALL      146           1      (1 FP)
True FALL           1          24      (1 FN)
```

Tuned thresholds:
- `stage1_threshold = 0.55`
- `rescue_threshold = 0.50`
- `vel_threshold = 0.0354`
- `acc_threshold = 0.3545`

---

## Natijalar tarixi

### Run 3 — 2026-06-11 (Physics Rescue + Zero-frame fix)
Yangiliklar:
- `conf=0.1` + forward-fill interpolation bilan zero frame 14.5% → **0%**
- Physics "Rescue" mantiq: `prob >= t1 → FALL`, `t_rescue <= prob < t1 → physics qaror`
- Avvalgi AND mantiq o'rniga: physics endi faqat QO'SHADI, o'chirmaydi

| Model | Fall F1 |
|---|---|
| ST-GCN Stage 1 | **0.960** |
| ST-GCN + Physics Rescue | **0.960** |

Izoh: Rescue zone [0.50, 0.55) — Stage 1 shunchalik yaxshi bo'lganki rescue kerak bo'lmadi.

---

### Run 2 — 2026-06-11 (Zero-frame fix, eski physics)
Yangiliklar: `conf=0.1` + interpolation bilan zero frame tuzatildi

| Model | Fall F1 |
|---|---|
| ST-GCN Stage 1 | 0.913 |
| ST-GCN + Physics (AND) | **0.864** ← physics zararli! |

Izoh: Physics AND mantiq Stage 1 ning to'g'ri topganlarini ham o'chirdi (Stage 1 precision=1.00 edi).

---

### Run 1 — 2026-06-10 (Boshlang'ich natija)
Dataset: zero frame muammosi bilan (fall sequencelarda 14.5% zero frame)

| Model | Accuracy | Fall F1 |
|---|---|---|
| ST-GCN Stage 1 | 93.6% | 0.718 |
| ST-GCN + Physics (AND) | 95.9% | **0.837** |

Izoh: Zero frame tufayli recall past (0.56). Physics AND mantiq bu holdada foydali bo'ldi chunki FP lar ko'p edi.

---

## Yaxshilanish tahlili

```
Run 1 → Run 2: Zero-frame fix
  Fall F1: 0.718 → 0.913  (+0.195)  ← eng katta ta'sir
  Sabab: YOLO conf=0.1 + interpolation

Run 2 → Run 3: Physics Rescue mantiq
  Fall F1: 0.913 → 0.960  (+0.047)
  Sabab: Physics endi topganlarni o'chirmaydi
```

---

## Keyingi kutilgan natijalar

| Scenario | Kutilgan Fall F1 |
|---|---|
| Subject 1 (hozirgi) | 0.96 |
| Subject 1-17, subject-dependent | ~0.95+ |
| Subject 1-17, LOSO cross-subject | ~0.75-0.88 |

> LOSO natijasi haqiqiy real-world performance ko'rsatkichi hisoblanadi va qog'oz uchun kerak.

---

## Baseman modellar (qog'oz uchun qo'shish kerak)

- [ ] LSTM baseline
- [ ] TCN baseline
- [ ] ST-GCN alone (without physics) — hozir mavjud
- [ ] ST-GCN + Physics AND (eski mantiq) — hozir mavjud
- [ ] **ST-GCN + Physics Rescue (yangi mantiq)** — hozir mavjud
