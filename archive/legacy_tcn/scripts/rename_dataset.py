import os
import re
from pathlib import Path

# !!! Loyihangdagi dataset yo‘lini moslab yoz
DATASET_ROOT = r"C:\Users\ali\Projects\FALL_RESEARCH\data\dataset"

# Activity raqamlarini oson nomga o‘tkazamiz
activity_map = {
    "Activity1":  "front_fall",
    "Activity2":  "front_fall",
    "Activity3":  "back_fall",
    "Activity4":  "side_fall",
    "Activity5":  "back_fall",
    "Activity6":  "walk",
    "Activity7":  "stand",
    "Activity8":  "sit_table",
    "Activity9":  "bend",
    "Activity10": "jump",
}

# Masalan: Subject1Activity3Trial1Camera2
pattern = re.compile(
    r"^Subject\d+Activity(?P<activity>\d+)Trial\d+Camera(?P<camera>\d+)$"
)

def zero_pad_index(idx: int, total_digits: int = 6) -> str:
    """1 -> 000001"""
    return str(idx).zfill(total_digits)

def rename_images_in_folder(folder_path: Path):
    """Har bir papkadagi barcha .png fayllarni tartib bilan qayta nomlaydi."""
    image_exts = [".png"]
    images = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
    images.sort(key=lambda p: p.name)

    for i, img_path in enumerate(images, start=1):
        new_name = f"img_{zero_pad_index(i)}{img_path.suffix.lower()}"
        new_path = folder_path / new_name
        if img_path.name != new_name:
            img_path.rename(new_path)

    print(f"[OK] {folder_path.name}: {len(images)} frames renamed")

def main():
    root = Path(DATASET_ROOT)

    for old_folder in list(root.iterdir()):
        if not old_folder.is_dir():
            continue

        m = pattern.match(old_folder.name)
        if not m:
            print(f"[SKIP] {old_folder.name} - pattern mos kelmadi")
            continue

        activity_id = m.group("activity")   # masalan "3"
        camera_id   = m.group("camera")     # masalan "2"

        activity_key = f"Activity{activity_id}"
        if activity_key not in activity_map:
            print(f"[WARN] {activity_key} mapping topilmadi, o'tkazib yuborildi")
            continue

        activity_label = activity_map[activity_key]
        new_folder_name = f"Activity{activity_id}_{activity_label}_cam{camera_id}"
        new_folder_path = root / new_folder_name

        # 1️⃣ Rasmlar nomini tartiblash
        rename_images_in_folder(old_folder)

        # 2️⃣ Papka nomini o‘zgartirish
        if old_folder.name != new_folder_name:
            if new_folder_path.exists():
                print(f"[ERROR] {new_folder_name} allaqachon mavjud, o'tkazib yuborildi.")
            else:
                old_folder.rename(new_folder_path)
                print(f"[OK] {old_folder.name} → {new_folder_name}")

if __name__ == "__main__":
    main()
