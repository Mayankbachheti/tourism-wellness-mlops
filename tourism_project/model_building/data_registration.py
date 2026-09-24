# Register the raw tourism.csv to the Hugging Face Hub as a versioned dataset
import os
from huggingface_hub import HfApi, create_repo

DATASET_REPO = "BigBachheti/tourism-wellness-dataset"
LOCAL_FILE = "tourism_project/data/tourism.csv"

def main():
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"Dataset repo '{DATASET_REPO}' already exists — uploading a new version.")
    except Exception:
        print(f"Dataset repo '{DATASET_REPO}' not found — creating it.")
        create_repo(repo_id=DATASET_REPO, repo_type="dataset", token=token, private=False)

    api.upload_file(
        path_or_fileobj=LOCAL_FILE,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=token,
    )
    print("Raw dataset registered on the Hugging Face Hub:", DATASET_REPO)

if __name__ == "__main__":
    main()
