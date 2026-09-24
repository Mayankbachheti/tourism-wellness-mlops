# Downloading the train and test data from Hugging Face,
# trains different models using MLflow tunes the best model and uploads the final model to Hugging Face.
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_REPO = "BigBachheti/tourism-wellness-dataset"
MODEL_REPO = "BigBachheti/tourism-wellness-model"


def load_data(token):
    train_path = hf_hub_download(repo_id=DATASET_REPO, filename="train.csv", repo_type="dataset", token=token)
    test_path = hf_hub_download(repo_id=DATASET_REPO, filename="test.csv", repo_type="dataset", token=token)
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return preprocessor


def main():
    token = os.environ.get("HF_TOKEN")
    train_df, test_df = load_data(token)

    target = "ProdTaken"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    mlflow.set_experiment("tourism-wellness-package-prediction")

    best_name, best_f1, best_pipe = None, -1, None
    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")

            print(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1, best_name, best_pipe = f1, name, pipe

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    # Hyperparameter tuning of the selected model family
    param_dist = None
    if best_name == "RandomForest":
        param_dist = {
            "clf__n_estimators": [100, 200, 300, 400],
            "clf__max_depth": [4, 6, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        }
    elif best_name == "XGBoost":
        param_dist = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.8, 1.0],
        }
    elif best_name == "DecisionTree":
        param_dist = {
            "clf__max_depth": [3, 5, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }

    if param_dist:
        with mlflow.start_run(run_name=f"{best_name}_tuned"):
            search = RandomizedSearchCV(best_pipe, param_dist, n_iter=20, scoring="f1", cv=5, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            preds = best_pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(
    best_pipe,
    artifact_path="model",
    registered_model_name="tourism-wellness-best-model",
    serialization_format="cloudpickle",
)

            print(f"Tuned {best_name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
            print("Best params:", search.best_params_)

    # Save and upload the final pipeline
    os.makedirs("tourism_project/model_building/output", exist_ok=True)
    model_path = "tourism_project/model_building/output/best_model_v1.joblib"
    joblib.dump(best_pipe, model_path)

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    except Exception:
        create_repo(repo_id=MODEL_REPO, repo_type="model", token=token, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model_v1.joblib",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token,
    )
    print("Best model uploaded to the Hugging Face Hub:", MODEL_REPO)


if __name__ == "__main__":
    main()
