from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_DIR

EXPECTED_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}


def _read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}. Run: python -m src.download_data")
    return pd.read_csv(path, parse_dates=parse_dates)


def load_raw_data(raw_dir: Path = DATA_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all raw CSV files with timestamp parsing."""
    data = {
        "telemetry": _read_csv(raw_dir / EXPECTED_FILES["telemetry"], parse_dates=["datetime"]),
        "errors": _read_csv(raw_dir / EXPECTED_FILES["errors"], parse_dates=["datetime"]),
        "maint": _read_csv(raw_dir / EXPECTED_FILES["maint"], parse_dates=["datetime"]),
        "failures": _read_csv(raw_dir / EXPECTED_FILES["failures"], parse_dates=["datetime"]),
        "machines": _read_csv(raw_dir / EXPECTED_FILES["machines"]),
    }

    for key, frame in data.items():
        frame.columns = [c.strip() for c in frame.columns]
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")

    return data
