import yaml
import os

# CONFIG
yaml_path = r"C:\Users\ali\Projects\fall_research\ppe\data\data.yaml"
data_root = r"C:\Users\ali\Projects\fall_research\ppe\data"
class_to_remove = "fire"
class_id_to_remove = 3  # fire = 3

def remove_fire_from_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if class_to_remove in data["names"]:
        data["names"].remove(class_to_remove)
        print("🔥 Removed from YAML classes.")

    data["nc"] = len(data["names"])

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print("✅ YAML updated:", data)


def clean_labels_and_images(split):
    labels_path = os.path.join(data_root, split, "labels")
    images_path = os.path.join(data_root, split, "images")

    remove_list = []  # images to delete

    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue

        label_fp = os.path.join(labels_path, label_file)

        with open(label_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # filtering out fire class lines
        new_lines = []
        contains_fire = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():
                cid = int(parts[0])
                if cid == class_id_to_remove:
                    contains_fire = True
                    continue
            new_lines.append(line)

        # overwrite if needed
        if contains_fire:
            print(f"🧹 Removing fire labels from: {label_file}")

            if new_lines:
                # Write cleaned labels
                with open(label_fp, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            else:
                # If file becomes empty → remove label + image
                os.remove(label_fp)
                img_name = label_file.replace(".txt", ".jpg")
                remove_list.append(img_name)

    # delete images that became class-empty
    for img in remove_list:
        img_fp = os.path.join(images_path, img)
        if os.path.exists(img_fp):
            os.remove(img_fp)
            print(f"🗑️ Deleted unused image: {img}")

    print(f"✅ {split} split cleaned.")


if __name__ == "__main__":
    print("========== FIRE CLASS CLEANER ==========")
    remove_fire_from_yaml(yaml_path)
    clean_labels_and_images("train")
    clean_labels_and_images("val")
    print("=========== DONE ===========")
