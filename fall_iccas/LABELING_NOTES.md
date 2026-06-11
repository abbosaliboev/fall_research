# UP-Fall Dataset — Avtomatik Labeling

## Nima qilindi?

`label_dataset.py` skripti yozildi. Bu skript UP-Fall datasetidagi barcha CSV fayllarni o'qib, har bir qatorga avtomatik `label` ustuni qo'shadi (`1` = yiqilish, `0` = normal harakat) va hammasini bitta `labeled_dataset.csv` faylga birlashtiradi.

---

## Dataset strukturasi

```
dataset/
  SubjectN/
    ActivityN/
      TrialN/
        SubjectNActivityNTrialN.csv
```

Har bir CSV faylida 2 ta header qator bor (birinchisi — sensor guruhi nomlari, ikkinchisi — o'q nomlari), keyin sensor ma'lumotlari keladi. Har bir qatorda 47 ustun bor (dokumentatsiyada 46 ta deyilgan, lekin aslida bitta qo'shimcha ustun ham mavjud).

---

## Labeling mantigi

UP-Fall datasetida 11 ta activity bor. Maqola asosida ularni ikki guruhga bo'lish mumkin:

| Activity | Nomi | Label |
|---|---|---|
| 1 | Falling forward (hands) | **1** — FALL |
| 2 | Falling forward (knees) | **1** — FALL |
| 3 | Falling sideways | **1** — FALL |
| 4 | Falling backward | **1** — FALL |
| 5 | Hitting obstacle while walking | **1** — FALL |
| 6 | Sitting abruptly | **0** — NO-FALL |
| 7 | Walking | **0** — NO-FALL |
| 8 | Standing | **0** — NO-FALL |
| 9 | Sitting | **0** — NO-FALL |
| 10 | Picking up an object | **0** — NO-FALL |
| 11 | Jumping | **0** — NO-FALL |

**Nega shunday?**
Label papka nomidan (ActivityN) aniqlanadi, CSV ichidagi ustunlardan emas. Bu ishonchli, chunki papka nomi dataset egasi tomonidan belgilangan — CSV ichidagi ustunlarni o'qish shart emas.

---

## CSV o'qishdagi muammo va yechim

**Muammo:** Pandas `header=[0,1]` bilan o'qishda xato berdi, chunki ikkala header qatordagi ustunlar soni mos kelmaydi (birinchi qatorda merged cell-lar bor).

**Yechim:** `skiprows=2` ishlatildi — ikkala header qator o'tkazib yuborildi va ustun nomlari qo'lda belgilandi (`COLUMN_NAMES` ro'yxati orqali).

---

## Chiqish fayli — `labeled_dataset.csv`

Subject1 uchun natija:

| | Soni |
|---|---|
| Jami qatorlar | 17,932 |
| FALL qatorlar | 2,832 |
| NO-FALL qatorlar | 15,100 |

Fayldagi asosiy ustunlar:
- `ankle_acc_x/y/z`, `ankle_gyr_x/y/z`, `ankle_lux`
- `pocket_*`, `belt_*`, `neck_*`, `wrist_*` (bir xil format)
- `brain`, `ir1`–`ir6`
- `subject_id`, `activity_id`, `trial_id`, `activity_name`
- **`label`** — maqsad ustun (0 yoki 1)

---

## Keyingi subjectlar uchun

Boshqa subjectlar yuklangandan keyin ularni `dataset/SubjectN/` papkasiga solib, skriptni qayta ishga tushirish kifoya. Skript avtomatik barcha subjectlarni topib, bitta katta CSV ga birlashtiradi.

```bash
python label_dataset.py
```

---

*Manba: Lourdes Martínez-Villaseñor et al., "UP-Fall Detection Dataset: A Multimodal Approach", Sensors 19(9), 1988, 2019.*

---

## CV Dataset (kamera rasmlari uchun)

Sensor CSV lari emas, kamera rasmlari asosida CV model uchun dataset:

```
cv_dataset/
  X.npy    # (N, 30, 17, 3) — N ta window, 30 frame, 17 joint, [x, y, conf]
  y.npy    # (N,) — 0/1 labels
  meta.csv # subject, activity, trial, start_frame
```

`prepare_cv_dataset.py` ishlatiladi. Labeling mantigi bir xil — activity papka nomidan.

### Hal qilingan muammolar

**Camera papka nomi xatosi:**
Dataset da barcha `Subject1/ActivityN/TrialN/` ichidagi kamera papkasi `Subject1Activity2TrialXCameraX` deb nomlangan edi (hammasi Activity2). PowerShell script bilan parent papkadagi activity raqamidan foydalanib 54 ta papka to'g'rilandi.

**YOLO zero frame muammosi:**
Yiqilish paytida YOLO odamni topa olmay `[0, 0, 0]` keypoint qo'yardi. Fall sequencelarda bu 14.5% edi (Activity 2 — orqaga yiqilish — ayniqsa yomon, ba'zi windowlarda 100% zero).

Yechim:
1. `conf=0.1` — past threshold bilan ko'proq holat aniqlansin
2. `interpolate_zero_frames()` — zero frame larga avvalgi aniqlanganidan foydalanish

Natija: **0% zero frame** (14.5% dan)
