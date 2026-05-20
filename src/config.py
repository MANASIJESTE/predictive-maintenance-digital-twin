from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_SLUG = "arnabbiswas1/microsoft-azure-predictive-maintenance"
RANDOM_STATE = 42
LABEL_HORIZON_HOURS = 24
ROLLING_WINDOWS_HOURS = [3, 12, 24]

MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "features.parquet"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TRACKING_PATH = REPORTS_DIR / "tracking_runs.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.csv"
