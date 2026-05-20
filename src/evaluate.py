import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
import seaborn as sns

from src.config import (
    FEATURES_PATH,
    MODEL_PATH,
    METADATA_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    PREDICTIONS_PATH,
    DRIFT_REPORT_PATH,
)
from src.train import time_split


def false_alarms_per_machine_month(test_df: pd.DataFrame, y_true, y_pred) -> float:
    false_alarms = int(((y_pred == 1) & (y_true == 0)).sum())
    min_time = test_df["datetime"].min()
    max_time = test_df["datetime"].max()
    months = max((max_time - min_time).days / 30.0, 1 / 30.0)
    machines = test_df["machineID"].nunique()
    return false_alarms / (machines * months)


def make_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_columns:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std() or 1.0
        test_mean = test_df[col].mean()
        z_shift = abs(test_mean - train_mean) / train_std
        rows.append(
            {
                "feature": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "mean_shift_in_train_std": z_shift,
                "flag": "review" if z_shift > 1 else "ok",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_shift_in_train_std", ascending=False)


def evaluate_model() -> dict:
    features = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    feature_columns = metadata["feature_columns"]
    cutoff = pd.Timestamp(metadata["cutoff"])
    threshold = float(metadata["threshold"])

    train_df, test_df = time_split(features, cutoff)
    X_test = test_df[feature_columns]
    y_test = test_df["failure_24h"].astype(int).to_numpy()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities) if len(set(y_test)) > 1 else 0.0
    fa_rate = false_alarms_per_machine_month(test_df, y_test, predictions)

    metrics = {
        "threshold": threshold,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "false_alarms_per_machine_month": fa_rate,
        "alarm_definition": "An alarm is predicted when failure_risk_24h >= selected threshold for a machine-hour row.",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    pred_df = test_df[["machineID", "datetime", "failure_24h", "future_failure_component"]].copy()
    pred_df["failure_risk_24h"] = probabilities
    pred_df["prediction"] = predictions
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve, PR-AUC={pr_auc:.3f}")
    plt.savefig(REPORTS_DIR / "precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve, ROC-AUC={roc_auc:.3f}")
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_columns[i] for i in indices]
        top_importance = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", bbox_inches="tight")
        plt.close()

    # Metrics summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [precision, recall, f1]
    ax1.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylim([0, 1])
    ax1.set_title('Classification Metrics at Threshold')
    ax1.set_ylabel('Score')
    for i, v in enumerate(metrics_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    auc_names = ['ROC-AUC', 'PR-AUC']
    auc_values = [roc_auc, pr_auc]
    ax2.bar(auc_names, auc_values, color=['#d62728', '#9467bd'])
    ax2.set_ylim([0, 1])
    ax2.set_title('Area Under Curve Metrics')
    ax2.set_ylabel('Score')
    for i, v in enumerate(auc_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Greens', cbar=False, ax=ax3,
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix (with normalized %)')
    
    dataset_stats = {
        'Train Samples': len(train_df),
        'Test Samples': len(test_df),
        'Failure Rate': f"{(y_test.sum() / len(y_test) * 100):.1f}%"
    }
    ax4.axis('off')
    stats_text = "Dataset Statistics\n" + "\n".join([f"{k}: {v}" for k, v in dataset_stats.items()])
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "metrics_summary.png", bbox_inches="tight", dpi=100)
    plt.close()

    drift = make_drift_report(train_df, test_df, feature_columns)
    drift.to_csv(DRIFT_REPORT_PATH, index=False)

    # Model comparison plot
    try:
        comparison_path = REPORTS_DIR / "model_comparison.csv"
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # PR-AUC comparison
            comparison_df_sorted = comparison_df.sort_values('test_pr_auc', ascending=True)
            ax1.barh(comparison_df_sorted['run_name'], comparison_df_sorted['test_pr_auc'], color='#9467bd')
            ax1.set_xlabel('PR-AUC')
            ax1.set_title('Model Comparison: PR-AUC')
            ax1.set_xlim([0, 1])
            for i, v in enumerate(comparison_df_sorted['test_pr_auc']):
                ax1.text(v + 0.02, i, f'{v:.3f}', va='center')
            
            # ROC-AUC comparison
            comparison_df_sorted = comparison_df.sort_values('test_roc_auc', ascending=True)
            ax2.barh(comparison_df_sorted['run_name'], comparison_df_sorted['test_roc_auc'], color='#d62728')
            ax2.set_xlabel('ROC-AUC')
            ax2.set_title('Model Comparison: ROC-AUC')
            ax2.set_xlim([0, 1])
            for i, v in enumerate(comparison_df_sorted['test_roc_auc']):
                ax2.text(v + 0.02, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "model_comparison.png", bbox_inches="tight", dpi=100)
            plt.close()
            print(f"✓ Model Comparison: {REPORTS_DIR / 'model_comparison.png'}")
    except Exception as e:
        print(f"⚠ Could not generate model comparison plot: {e}")

    print(json.dumps(metrics, indent=2))
    print(f"✓ Saved metrics: {METRICS_PATH}")
    print(f"✓ Generated plots:")
    print(f"  - ROC Curve: {REPORTS_DIR / 'roc_curve.png'}")
    print(f"  - Precision-Recall Curve: {REPORTS_DIR / 'precision_recall_curve.png'}")
    print(f"  - Confusion Matrix: {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"  - Feature Importance: {REPORTS_DIR / 'feature_importance.png'}")
    print(f"  - Metrics Summary: {REPORTS_DIR / 'metrics_summary.png'}")
    print(f"✓ Saved drift report: {DRIFT_REPORT_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
