import os
from pathlib import Path

# =============================
# CONFIG
# =============================
ROOT = Path(r"C:\Users\ali\Projects\fall_research\unify_ppe\data")

LABEL_DIRS = [
    ROOT / "train" / "labels",
    ROOT / "val" / "labels"
]

IMG_DIRS = [
    ROOT / "train" / "images",
    ROOT / "val" / "images"
]

FIRE_CLASS_ID = "3"   # fire class index (in your YAML)


# =============================
# HELPERS
# =============================
def has_fire(label_path):
    """Check if a label file contains fire class."""
    with open(label_path, "r") as f:
        for line in f:
            cls = line.split()[0]
            if cls == FIRE_CLASS_ID:
                return True
    return False


# =============================
# MAIN
# =============================
removed = []

for lbl_dir, img_dir in zip(LABEL_DIRS, IMG_DIRS):

    for txt in lbl_dir.glob("*.txt"):
        if has_fire(txt):
            # matching image
            img_name = txt.stem

            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = img_dir / f"{img_name}{ext}"
                if img_path.exists():
                    os.remove(img_path)

            os.remove(txt)  # delete label
            removed.append(txt.stem)

# =============================
# REPORT
# =============================
print("🔥 FIRE LABELS REMOVED =", len(removed))
for r in removed:
    print(" -", r)

print("\nDone.")
