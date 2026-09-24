# ------------------------------------------------------------------
# Downloads the raw dataset from the Hugging Face Hub, cleans it,
# splits it into train/test sets, and pushes the processed splits
# back to the same dataset repository.
# ------------------------------------------------------------------
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "BigBachheti/tourism-wellness-dataset"  # <-- update with your HF username

def main():
    token = os.environ.get("HF_TOKEN")

    raw_path = hf_hub_download(
        repo_id=DATASET_REPO, filename="tourism.csv", repo_type="dataset", token=token
    )
    df = pd.read_csv(raw_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Fix data-entry inconsistencies discovered during EDA
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df = df.drop(columns=["CustomerID"])

    target = "ProdTaken"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy(); train_df[target] = y_train
    test_df = X_test.copy(); test_df[target] = y_test

    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)

    api = HfApi(token=token)
    for fname in ["train.csv", "test.csv"]:
        api.upload_file(
            path_or_fileobj=f"tourism_project/data/{fname}",
            path_in_repo=fname,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=token,
        )

    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("Processed train/test data uploaded to the Hugging Face Hub:", DATASET_REPO)

if __name__ == "__main__":
    main()
