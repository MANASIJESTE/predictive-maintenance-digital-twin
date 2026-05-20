from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def time_split(df: pd.DataFrame, cutoff=None):
    """
    Backward-compatible function for evaluate.py.

    Splits data using one global time cutoff.
    Returns only train_df and test_df because evaluate.py expects 2 values.
    """

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    if cutoff is None:
        cutoff = df["datetime"].quantile(0.70)
    else:
        cutoff = pd.to_datetime(cutoff)

    train_df = df[df["datetime"] <= cutoff].copy()
    test_df = df[df["datetime"] > cutoff].copy()

    return train_df, test_df

def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select model features safely.

    Removes identifiers, target labels, future component information,
    and any leakage-like columns.
    """

    forbidden_exact = {
        "datetime",
        "machineID",
        "label",
        "failure_24h",
        "failure_component_24h",
        "future_failure_component",
        "failure",
        "component",
    }

    forbidden_tokens = [
        "failure",
        "label",
        "future",
        "component_24h",
    ]

    feature_columns = []

    for col in df.columns:
        col_str = str(col)

        if col_str in forbidden_exact:
            continue

        if any(token in col_str.lower() for token in forbidden_tokens):
            continue

        if df[col].dtype == "object":
            continue

        feature_columns.append(col_str)

    merge_artifacts = [
        col for col in feature_columns
        if "_x" in col or "_y" in col
    ]

    if merge_artifacts:
        raise ValueError(
            "Merge artifact columns found in features. "
            f"Clean feature engineering first: {merge_artifacts[:20]}"
        )

    leakage_columns = [
        col for col in feature_columns
        if any(token in col.lower() for token in forbidden_tokens)
    ]

    if leakage_columns:
        raise ValueError(
            f"Possible leakage columns found in model features: {leakage_columns}"
        )

    return feature_columns


def choose_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Choose threshold using best F1 score.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1))

    return float(thresholds[best_idx])


def evaluate_run(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    threshold = choose_threshold_by_f1(y_test, y_proba)

    return {
        "threshold": float(threshold),
        "test_pr_auc": float(pr_auc),
        "test_roc_auc": float(roc_auc),
    }


def train_models(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train baseline Logistic Regression and final Random Forest.
    Uses one global time-aware cutoff across all machines.
    """

    df = feature_df.copy()

    if "failure_24h" not in df.columns:
        raise ValueError("Expected label column 'failure_24h' not found.")

    cutoff = df["datetime"].quantile(0.70)
    train_df, test_df = time_split(df, cutoff)

    feature_columns = select_feature_columns(df)

    X_train = train_df[feature_columns]
    y_train = train_df["failure_24h"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["failure_24h"].astype(int)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Feature columns:", len(feature_columns))
    print("Cutoff:", cutoff)
    print("Train positive rate:", y_train.mean())
    print("Test positive rate:", y_test.mean())

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("predictive-maintenance-digital-twin")
    else:
        print("⚠️  MLflow not available. Skipping experiment tracking.")

    runs = []

    baseline_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    final_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gradient_boosting_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    models = [
        ("baseline_logistic_regression", baseline_model),
        ("final_random_forest", final_model),
        ("gradient_boosting", gradient_boosting_model),
    ]

    if XGBOOST_AVAILABLE:
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgboost_model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbosity=0,
        )
        models.append(("xgboost", xgboost_model))
    else:
        print("⚠️  XGBoost not installed. Install with: pip install xgboost")

    fitted_models = {}

    for run_name, model in models:
        if MLFLOW_AVAILABLE:
            mlflow_context = mlflow.start_run(run_name=run_name)
            mlflow_context.__enter__()
        
        model.fit(X_train, y_train)
        metrics = evaluate_run(model, X_test, y_test)

        if MLFLOW_AVAILABLE:
            mlflow.log_param("run_name", run_name)
            mlflow.log_param("cutoff", str(cutoff))
            mlflow.log_param("n_train", len(train_df))
            mlflow.log_param("n_test", len(test_df))
            mlflow.log_param("n_features", len(feature_columns))
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow_context.__exit__(None, None, None)

        run_result = {
            "run_name": run_name,
            "cutoff": str(cutoff),
            "threshold": metrics["threshold"],
            "test_pr_auc": metrics["test_pr_auc"],
            "test_roc_auc": metrics["test_roc_auc"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(feature_columns),
        }

        runs.append(run_result)
        fitted_models[run_name] = model

    tracking_df = pd.DataFrame(runs).sort_values("test_pr_auc", ascending=False)
    tracking_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(tracking_df[["run_name", "test_pr_auc", "test_roc_auc"]].to_string(index=False))
    print("="*70)

    final_threshold = float(
        tracking_df.loc[
            tracking_df["run_name"] == "final_random_forest",
            "threshold",
        ].iloc[0]
    )

    metadata = {
        "feature_columns": feature_columns,
        "cutoff": str(cutoff),
        "threshold": final_threshold,
        "label_definition": "failure_24h = 1 if any failure occurs in (t, t+24h], else 0",
        "final_model": "RandomForestClassifier",
        "baseline_model": "LogisticRegression",
    }

    joblib.dump(
        fitted_models["final_random_forest"],
        ARTIFACTS_DIR / "final_model.joblib",
    )

    joblib.dump(
        metadata,
        ARTIFACTS_DIR / "model_metadata.joblib",
    )

    tracking_df.to_csv(REPORTS_DIR / "tracking_runs.csv", index=False)

    with open(REPORTS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(tracking_df)
    print("Saved final model:", (ARTIFACTS_DIR / "final_model.joblib").resolve())
    print("Saved metadata:", (ARTIFACTS_DIR / "model_metadata.joblib").resolve())
    print("Saved tracking:", (REPORTS_DIR / "tracking_runs.csv").resolve())

    return tracking_df


def train_model(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper for existing pipeline.py.
    """
    return train_models(feature_df)


if __name__ == "__main__":
    features_path = ARTIFACTS_DIR / "features.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            "features.parquet not found. Run python -m src.pipeline first."
        )

    features = pd.read_parquet(features_path)
    train_models(features)