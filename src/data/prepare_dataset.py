import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def prepare_starmen_dataset(csv_path, test_split=0.3):
    """
    Prepare Starmen dataset:
    - Load the data from csv files.
    - Sanity check for missing datas or files.
    - Train-Test split
    - Write to files.
    """
    DATASET_NAME = "starmen"

    datas = pd.read_csv(csv_path)
    datas = datas.rename(
        columns={"t": "age", "tau": "baseline_age", "path": "img_path"}
    )

    # Change relative path
    DATA_DIR = csv_path.split("df.csv")[0]
    datas["img_path"] = datas["img_path"].apply(
        lambda x: os.path.join(DATA_DIR, "images", x.split("/images/")[-1])
    )

    # Get subject id
    def extract_subject_id(str):
        return str.split("__")[3].split("subject_s")[-1]

    datas["id"] = datas["id"].apply(extract_subject_id)

    # Sanity check
    sanity_check = dict()
    # check_null = datas.isnull().sum()
    # if check_null:
    #     sanity_check["null"] = datas[datas.isnull()]
    # else:
    #     sanity_check["nulll"] = None

    check_duplicates = datas.duplicated().sum()
    if check_duplicates:
        sanity_check["duplicates"] = datas[datas.duplicated()]
    else:
        sanity_check["duplicates"] = None

    # Check img path exists

    datas["path_exists"] = datas["img_path"].apply(os.path.exists)
    missing_paths = datas[~datas["path_exists"]]
    if missing_paths.empty:
        sanity_check["missing_paths"] = None
    else:
        sanity_check["missing_paths"] = missing_paths["img_path"].tolist()
        datas = datas[datas["path_exists"]].reset_index(drop=True)
    datas.drop(columns=["path_exists"], inplace=True)

    # Get list of subject_ids
    ids = datas["id"].unique()

    # Train Test split
    train_ids, temp_ids = train_test_split(ids, test_size=test_split, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_df = datas[datas["id"].isin(train_ids)].reset_index(drop=True)
    val_df = datas[datas["id"].isin(val_ids)].reset_index(drop=True)
    test_df = datas[datas["id"].isin(test_ids)].reset_index(drop=True)

    # Write to csv files
    train_file = os.path.join(DATA_DIR, f"{DATASET_NAME}_train.csv")
    test_file = os.path.join(DATA_DIR, f"{DATASET_NAME}_test.csv")
    val_file = os.path.join(DATA_DIR, f"{DATASET_NAME}_val.csv")
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("Finish preparing dataset")
    for d, f in zip((train_df, val_df, test_df), (train_file, val_file, test_file)):
        print(f"Train dataset: ({len(d['id'])} subjects) created at {f}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from CSV file")
    parser.add_argument("--csv_path", type=str, default="data/starmen/output_random_noacc/df.csv",
                        help="Path to the input CSV file")

    args = parser.parse_args()
    csv_path = args.csv_path

    assert os.path.exists(csv_path), f"Path does not exist: {csv_path}"
    assert csv_path.endswith(".csv"), "Please input a CSV file."

    if "starmen" in csv_path: 
        prepare_starmen_dataset(csv_path)
    else: 
        print("Dataset is not supported.")



if __name__ == "__main__":
    main()