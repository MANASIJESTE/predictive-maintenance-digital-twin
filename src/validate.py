import pandas as pd

REQUIRED_COLUMNS = {
    "telemetry": {"datetime", "machineID", "volt", "rotate", "pressure", "vibration"},
    "errors": {"datetime", "machineID", "errorID"},
    "maint": {"datetime", "machineID", "comp"},
    "failures": {"datetime", "machineID", "failure"},
    "machines": {"machineID", "model", "age"},
}


def validate_raw_data(data: dict[str, pd.DataFrame]) -> None:
    """Basic schema, null, duplicate, and timestamp checks."""
    for name, required_cols in REQUIRED_COLUMNS.items():
        if name not in data:
            raise ValueError(f"Missing table: {name}")

        missing_cols = required_cols - set(data[name].columns)
        if missing_cols:
            raise ValueError(f"{name} is missing columns: {sorted(missing_cols)}")

        if data[name].empty:
            raise ValueError(f"{name} is empty")

    for name in ["telemetry", "errors", "maint", "failures"]:
        if data[name]["datetime"].isna().any():
            raise ValueError(f"{name} contains invalid datetime values")

    if data["telemetry"].duplicated(["datetime", "machineID"]).any():
        raise ValueError("telemetry has duplicate rows for the same machineID and datetime")


def clip_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Deterministic IQR clipping for telemetry outliers."""
    output = df.copy()
    for col in columns:
        q1 = output[col].quantile(0.25)
        q3 = output[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        output[col] = output[col].clip(lower, upper)
    return output
