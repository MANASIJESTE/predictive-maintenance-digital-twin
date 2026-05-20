from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train import select_feature_columns, time_split


ARTIFACTS_DIR = Path("artifacts")


def train_and_score(X_train, y_train, X_test, y_test, name: str):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name}")
    print("-" * len(name))
    print("PR-AUC:", round(pr_auc, 6))
    print("ROC-AUC:", round(roc_auc, 6))

    return pr_auc, roc_auc


def main():
    features_path = ARTIFACTS_DIR / "features.parquet"
    metadata_path = ARTIFACTS_DIR / "model_metadata.joblib"

    df = pd.read_parquet(features_path)
    metadata = joblib.load(metadata_path)

    cutoff = metadata["cutoff"]

    train_df, test_df = time_split(df, cutoff)

    feature_columns = select_feature_columns(df)

    X_train = train_df[feature_columns]
    y_train = train_df["failure_24h"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["failure_24h"].astype(int)

    print("\n=== Dataset sanity ===")
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Number of features:", len(feature_columns))
    print("Train positive rate:", round(y_train.mean(), 6))
    print("Test positive rate:", round(y_test.mean(), 6))

    # 1. Normal model
    train_and_score(
        X_train,
        y_train,
        X_test,
        y_test,
        "Normal labels",
    )

    # 2. Shuffled-label sanity test
    rng = np.random.default_rng(42)
    y_train_shuffled = y_train.copy().to_numpy()
    rng.shuffle(y_train_shuffled)

    train_and_score(
        X_train,
        y_train_shuffled,
        X_test,
        y_test,
        "Shuffled training labels",
    )

    # 3. Telemetry-only model
    telemetry_cols = [
        col for col in feature_columns
        if any(sensor in col for sensor in ["volt", "rotate", "pressure", "vibration"])
    ]

    train_and_score(
        X_train[telemetry_cols],
        y_train,
        X_test[telemetry_cols],
        y_test,
        "Telemetry-only features",
    )

    # 4. Error-only model
    error_cols = [
        col for col in feature_columns
        if "error" in col.lower()
    ]

    if error_cols:
        train_and_score(
            X_train[error_cols],
            y_train,
            X_test[error_cols],
            y_test,
            "Error-only features",
        )
    else:
        print("\nNo error columns found.")

    # 5. Maintenance-only model
    maintenance_cols = [
        col for col in feature_columns
        if "maint" in col.lower()
    ]

    if maintenance_cols:
        train_and_score(
            X_train[maintenance_cols],
            y_train,
            X_test[maintenance_cols],
            y_test,
            "Maintenance-only features",
        )
    else:
        print("\nNo maintenance columns found.")

    print("\n=== Interpretation ===")
    print(
        "If shuffled-label PR-AUC drops close to the test positive rate, "
        "there is no obvious direct leakage."
    )
    print(
        "If shuffled-label PR-AUC is still very high, there is likely leakage."
    )


if __name__ == "__main__":
    main()