from pathlib import Path
import joblib
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")


def find_feature_columns(df: pd.DataFrame, metadata: dict) -> list[str]:
    print("\n=== Metadata keys ===")

    if isinstance(metadata, dict):
        print(list(metadata.keys()))
    else:
        print("Metadata is not a dictionary. Type:", type(metadata))

    possible_keys = [
        "feature_columns",
        "feature_cols",
        "features",
        "model_features",
        "input_features",
    ]

    if isinstance(metadata, dict):
        for key in possible_keys:
            if key in metadata:
                print(f"\nUsing feature list from metadata key: {key}")
                return metadata[key]

    print("\nNo feature list found in metadata.")
    print("Inferring feature columns from feature table.")

    drop_cols = [
        "datetime",
        "machineID",
        "label",
        "failure_24h",
        "failure_component_24h",
        "future_failure_component",
        "failure",
        "component",
    ]

    feature_cols = [
        col for col in df.columns
        if col not in drop_cols and df[col].dtype != "object"
    ]

    return feature_cols


def main():
    features_path = ARTIFACTS_DIR / "features.parquet"
    metadata_path = ARTIFACTS_DIR / "model_metadata.joblib"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found. Run python -m src.pipeline first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run python -m src.pipeline first."
        )

    df = pd.read_parquet(features_path)
    metadata = joblib.load(metadata_path)

    feature_cols = find_feature_columns(df, metadata)

    print("\n=== Basic feature table info ===")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    label_col = None

    if "failure_24h" in df.columns:
        label_col = "failure_24h"
    elif "label" in df.columns:
        label_col = "label"

    if label_col:
        print("Label column:", label_col)
        print("Positive label rate:", df[label_col].mean())
        print("\nLabel counts:")
        print(df[label_col].value_counts())
    else:
        print("WARNING: no label column found.")

    print("\n=== Checking suspicious feature names ===")

    forbidden_tokens = [
        "label",
        "failure",
        "future",
        "component_24h",
    ]

    suspicious = [
        col for col in feature_cols
        if any(token in str(col).lower() for token in forbidden_tokens)
    ]

    if suspicious:
        print("Possible leakage columns found in model features:")
        for col in suspicious:
            print("-", col)
    else:
        print("No direct leakage column names found in model features.")

    print("\n=== Checking _x/_y merge artifact columns ===")

    merge_artifacts = [
        col for col in feature_cols
        if "_x" in str(col) or "_y" in str(col)
    ]

    if merge_artifacts:
        print("Merge artifact columns found:")
        for col in merge_artifacts[:80]:
            print("-", col)
    else:
        print("No _x/_y merge artifact columns found.")

    print("\n=== Feature columns used/inferred ===")
    print("Number of feature columns:", len(feature_cols))

    for col in feature_cols[:120]:
        print("-", col)

    print("\n=== Important column presence checks ===")

    for col in [
        "label",
        "failure_24h",
        "failure_component_24h",
        "future_failure_component",
        "failure",
        "datetime",
        "machineID",
    ]:
        print(f"{col} present in feature table:", col in df.columns)

    print("\n=== Columns containing failure/label/component/future ===")

    risky_table_cols = [
        col for col in df.columns
        if "label" in str(col).lower()
        or "failure" in str(col).lower()
        or "component" in str(col).lower()
        or "future" in str(col).lower()
    ]

    if risky_table_cols:
        for col in risky_table_cols:
            print("-", col)
    else:
        print("None found.")

    print("\n=== Final diagnostic summary ===")

    if suspicious:
        print("Status: WARNING - possible leakage columns in model features.")
    elif merge_artifacts:
        print("Status: WARNING - merge artifact columns in model features.")
    else:
        print("Status: OK - no obvious leakage or merge artifacts in model features.")


if __name__ == "__main__":
    main()