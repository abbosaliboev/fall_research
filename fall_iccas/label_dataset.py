import os
import pandas as pd
import re

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "labeled_dataset.csv")

# Activities 1-5 = fall, 6-11 = no-fall
FALL_ACTIVITIES = {1, 2, 3, 4, 5}

ACTIVITY_NAMES = {
    1: "Falling forward (hands)",
    2: "Falling forward (knees)",
    3: "Falling sideways",
    4: "Falling backward",
    5: "Hitting obstacle",
    6: "Sitting abruptly",
    7: "Walking",
    8: "Standing",
    9: "Sitting",
    10: "Picking up object",
    11: "Jumping",
}


def get_csv_files(root_dir):
    records = []
    for subject_dir in sorted(os.listdir(root_dir)):
        subject_match = re.match(r"Subject(\d+)", subject_dir)
        if not subject_match:
            continue
        subject_id = int(subject_match.group(1))
        subject_path = os.path.join(root_dir, subject_dir)

        for activity_dir in sorted(os.listdir(subject_path)):
            activity_match = re.match(r"Activity(\d+)", activity_dir)
            if not activity_match:
                continue
            activity_id = int(activity_match.group(1))
            activity_path = os.path.join(subject_path, activity_dir)

            for trial_dir in sorted(os.listdir(activity_path)):
                trial_match = re.match(r"Trial(\d+)", trial_dir)
                if not trial_match:
                    continue
                trial_id = int(trial_match.group(1))
                trial_path = os.path.join(activity_path, trial_dir)

                for fname in os.listdir(trial_path):
                    if fname.endswith(".csv"):
                        records.append({
                            "path": os.path.join(trial_path, fname),
                            "subject": subject_id,
                            "activity": activity_id,
                            "trial": trial_id,
                        })
    return records


COLUMN_NAMES = [
    "timestamp",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyr_x", "ankle_gyr_y", "ankle_gyr_z",
    "ankle_lux",
    "pocket_acc_x", "pocket_acc_y", "pocket_acc_z",
    "pocket_gyr_x", "pocket_gyr_y", "pocket_gyr_z",
    "pocket_lux",
    "belt_acc_x", "belt_acc_y", "belt_acc_z",
    "belt_gyr_x", "belt_gyr_y", "belt_gyr_z",
    "belt_lux",
    "neck_acc_x", "neck_acc_y", "neck_acc_z",
    "neck_gyr_x", "neck_gyr_y", "neck_gyr_z",
    "neck_lux",
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "wrist_gyr_x", "wrist_gyr_y", "wrist_gyr_z",
    "wrist_lux",
    "brain",
    "ir1", "ir2", "ir3", "ir4", "ir5", "ir6",
    "csv_subject", "csv_activity", "csv_trial",
    "extra",  # undocumented extra column present in the files
]


def load_csv(path):
    # Skip the 2 header rows; use our own column names
    df = pd.read_csv(path, skiprows=2, header=None, low_memory=False)
    # Assign names only up to the actual number of columns
    names = COLUMN_NAMES[: len(df.columns)]
    df.columns = names
    return df


def main():
    records = get_csv_files(DATASET_DIR)
    if not records:
        print(f"No CSV files found under {DATASET_DIR}")
        return

    all_frames = []
    for rec in records:
        try:
            df = load_csv(rec["path"])
        except Exception as e:
            print(f"  [SKIP] {rec['path']}: {e}")
            continue

        df["subject_id"] = rec["subject"]
        df["activity_id"] = rec["activity"]
        df["trial_id"] = rec["trial"]
        df["activity_name"] = ACTIVITY_NAMES.get(rec["activity"], "Unknown")
        df["label"] = 1 if rec["activity"] in FALL_ACTIVITIES else 0  # 1=fall, 0=no-fall

        all_frames.append(df)
        lbl = "FALL" if rec["activity"] in FALL_ACTIVITIES else "NO-FALL"
        print(f"  S{rec['subject']} A{rec['activity']} T{rec['trial']} -> {lbl} ({len(df)} rows)")

    if not all_frames:
        print("No data loaded.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Total rows : {len(combined)}")
    print(f"FALL rows  : {(combined['label'] == 1).sum()}")
    print(f"NO-FALL rows: {(combined['label'] == 0).sum()}")
    print("\nPer-activity row counts:")
    print(combined.groupby(["activity_id", "activity_name", "label"]).size().to_string())


if __name__ == "__main__":
    main()
