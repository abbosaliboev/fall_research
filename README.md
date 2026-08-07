# Fall Research — Human Activity / Fall Detection

Bir necha yillik fall-detection tadqiqotlarini o'z ichiga olgan repo. Faol
loyihalar tepada, oldingi avlodlar `archive/` da saqlanadi.

## Papka tuzilmasi

- **`fall_iccas/`** — eng so'nggi tugallangan loyiha: ST-GCN + Physics Filter
  ikki bosqichli fall detection, UP-Fall dataset asosida (ICCAS 2026 maqolasi
  uchun). Batafsil: `fall_iccas/README.md`, `fall_iccas/RESULTS.md`.
- **`Fall_Research_Paper/`** — navbatdagi maqola/tadqiqot uchun yangi, faol
  ish joyi.
- **`archive/`** — oldingi avlodlar, tarixiy/qayta ishlatish uchun saqlangan:
  - `archive/2026_Fall_research/` — YOLO11-pose (10kp va full-kp) + TCN
    yondashuvi, dataset, model checkpointlar va training natijalari bilan.
  - `archive/legacy_tcn/` — eng birinchi TCN asosidagi versiya (`scripts/`,
    sequence/label CSV fayllari).
  - `archive/experiments_fall/` — eski checkpoint fayllari (FD-01, FD-02).
- **`PPT/`** — taqdimot materiallari.
- **`venv/`** — Python virtual environment (Git tomonidan e'tiborga
  olinmaydi).

> Eslatma: PPE (himoya kiyimi) detection loyihasi fall detection bilan
> aloqasi yo'qligi sababli `F:\Project_F\ppe` ga chiqarib qo'yilgan.

## Dataset

Dataset link:
https://sites.google.com/up.edu.mx/har-up/

Loyihaga qarab kerakli dataset shu loyiha papkasi ichidagi `data/` yoki
`dataset/` katalogiga joylashtiriladi. Katta xom datasetlarni Git'ga
commit qilmang.

## Setup

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ishlash

Har bir loyiha (`fall_iccas/`, `Fall_Research_Paper/`, `archive/...`) o'z
ichida mustaqil — kerakli scriptlar va o'qish uchun shu papkalardagi
README/RESULTS fayllariga qarang.

## Ignoring dataset files (important)

`.gitignore` allaqachon `data/`, `models/`, checkpoint va katta binary
fayllarni chiqarib tashlaydi. Agar oldin commit qilingan bo'lsa:

```powershell
git rm -r --cached "path/to/data"
git commit -m "Stop tracking dataset files"
```
