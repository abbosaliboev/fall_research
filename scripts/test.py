# check_splits.py
import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\ali\Projects\fall_research"
TRAIN_CSV = os.path.join(PROJECT_ROOT, "sequences_train.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "sequences_val.csv")
TEST_CSV  = os.path.join(PROJECT_ROOT, "sequences_test.csv")

def check_csv(path):
    df = pd.read_csv(path)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Total rows: {len(df)}")
    
    # Check NaNs in key columns
    print("\nNaN count per important column:")
    for col in ["subject", "activity", "clip", "label", "label_id"]:
        if col in df.columns:
            print(f"{col}: {df[col].isna().sum()}")

    # Check label distribution
    if "label" in df.columns:
        print("\nLabel counts:")
        print(df["label"].value_counts())

    if "label_id" in df.columns:
        print("\nLabel_id counts:")
        print(df["label_id"].value_counts())

check_csv(TRAIN_CSV)
check_csv(VAL_CSV)
check_csv(TEST_CSV)
