import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CSV_PATH = "data/Data_Entry_2017.csv"
OUT_DIR = "data/splits"
TARGET_LABEL = "Cardiomegaly"

def has_label(label_str: str, target: str) -> int:
    # "Cardiomegaly|Emphysema" -> 1, "No Finding" -> 0
    labels = str(label_str).split("|")
    return int(target in labels)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Clean column names a bit (some NIH CSVs have odd trailing spaces)
    df.columns = [c.strip() for c in df.columns]

    # Create binary target
    df["target"] = df["Finding Labels"].apply(lambda s: has_label(s, TARGET_LABEL))

    # Patient-wise split: train/val/test (80/10/10)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["Patient ID"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["Patient ID"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("Saved splits to:", OUT_DIR)
    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Positives (train):", train_df["target"].sum())

if __name__ == "__main__":
    main()