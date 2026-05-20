import argparse
import json

import joblib
import pandas as pd

from src.config import FEATURES_PATH, MODEL_PATH, METADATA_PATH, REPORTS_DIR
from src.digital_twin import make_digital_twin_response
from src.train import time_split


def predict(machine_id: int, timestamp: str) -> dict:
    features = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    feature_columns = metadata["feature_columns"]
    cutoff = pd.Timestamp(metadata["cutoff"])

    requested_time = pd.Timestamp(timestamp)
    exact = features[(features["machineID"] == machine_id) & (features["datetime"] == requested_time)]

    if exact.empty:
        machine_rows = features[features["machineID"] == machine_id].copy()
        if machine_rows.empty:
            raise ValueError(f"machineID={machine_id} does not exist in features")
        machine_rows["time_distance"] = (machine_rows["datetime"] - requested_time).abs()
        row_df = machine_rows.sort_values("time_distance").head(1).drop(columns=["time_distance"])
        used_timestamp = str(row_df.iloc[0]["datetime"])
    else:
        row_df = exact.head(1)
        used_timestamp = str(requested_time)

    row = row_df.iloc[0]
    risk = float(model.predict_proba(row_df[feature_columns])[:, 1][0])
    train_df, _ = time_split(features, cutoff)

    response = make_digital_twin_response(
        machine_id=int(row["machineID"]),
        timestamp=used_timestamp,
        risk=risk,
        row=row,
        train_reference=train_df,
        feature_columns=feature_columns,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description="Return predictive-maintenance digital twin JSON.")
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--timestamp", type=str, required=True, help="Example: 2015-10-01 08:00:00")
    args = parser.parse_args()

    response = predict(args.machine_id, args.timestamp)
    print(json.dumps(response, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "sample_prediction.json").write_text(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
