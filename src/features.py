from pathlib import Path
import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _to_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def add_rolling_telemetry_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling telemetry features per machine.

    Features:
    - mean over 3h, 12h, 24h
    - std over 3h, 12h, 24h
    - max over 3h, 12h, 24h

    These use current and previous rows only because the dataframe is sorted by
    machineID and datetime. In real-time prediction, telemetry at timestamp t is
    assumed to be available before prediction at timestamp t.
    """

    df = telemetry.copy()
    df = _to_datetime(df)
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    sensor_cols = ["volt", "rotate", "pressure", "vibration"]
    windows = [3, 12, 24]

    for col in sensor_cols:
        for window in windows:
            grouped = df.groupby("machineID")[col]

            # Create rolling features
            df[f"{col}_mean_{window}h_temp"] = (
                grouped
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{col}_std_{window}h_temp"] = (
                grouped
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

            df[f"{col}_max_{window}h_temp"] = (
                grouped
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

            # Shift by 72 hours to prevent leakage
            # Feature at time t uses sensor data from (t-72h-window to t-72h), not (t-window to t)
            df[f"{col}_mean_{window}h"] = df.groupby("machineID")[f"{col}_mean_{window}h_temp"].shift(72).fillna(0)
            df[f"{col}_std_{window}h"] = df.groupby("machineID")[f"{col}_std_{window}h_temp"].shift(72).fillna(0)
            df[f"{col}_max_{window}h"] = df.groupby("machineID")[f"{col}_max_{window}h_temp"].shift(72).fillna(0)

            # Drop temp columns
            df = df.drop(columns=[f"{col}_mean_{window}h_temp", f"{col}_std_{window}h_temp", f"{col}_max_{window}h_temp"])

    rolling_std_cols = [col for col in df.columns if "_std_" in col]
    df[rolling_std_cols] = df[rolling_std_cols].fillna(0)

    return df


def add_error_features(base_df: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean rolling error count features.

    This avoids repeated merge artifacts like:
    - error1_count_24h_x_x_x
    - error2_count_168h_y_y

    Final columns look like:
    - error1_count_24h
    - error1_count_168h
    - error2_count_24h
    - error2_count_168h
    """

    df = base_df.copy()
    errors = errors.copy()

    df = _to_datetime(df)
    errors = _to_datetime(errors)

    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # Build event matrix once
    error_events = (
        errors
        .assign(error_value=1)
        .pivot_table(
            index=["machineID", "datetime"],
            columns="errorID",
            values="error_value",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # Make column names normal strings
    error_events.columns = [str(col) for col in error_events.columns]

    # Rename raw event columns temporarily
    rename_map = {
        col: f"{col}_event"
        for col in error_events.columns
        if str(col).startswith("error")
    }

    error_events = error_events.rename(columns=rename_map)

    # Merge once only
    df = df.merge(
        error_events,
        on=["machineID", "datetime"],
        how="left",
    )

    event_cols = [col for col in df.columns if col.endswith("_event")]

    for col in event_cols:
        df[col] = df[col].fillna(0)

    # Rolling counts over 24 hours and 168 hours
    # SHIFT BY 72 HOURS to prevent leakage
    # At time t, use error counts from (t-96h to t-72h), NOT from (t-24h to t)
    for event_col in event_cols:
        base_name = event_col.replace("_event", "")

        for window in [24, 168]:
            rolling_col = f"{base_name}_count_{window}h_temp"
            df[rolling_col] = (
                df
                .groupby("machineID")[event_col]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            # Shift back 72 hours: feature at t uses data from (t-96h to t-72h)
            df[f"{base_name}_count_{window}h"] = (
                df
                .groupby("machineID")[rolling_col]
                .shift(72)  # Shift by 72 hours to remove leakage
                .fillna(0)
            )
            df = df.drop(columns=[rolling_col])

    # Drop raw event indicator columns after making rolling features
    df = df.drop(columns=event_cols)

    return df


def add_maintenance_features(base_df: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """
    Add hours since last maintenance per component.

    Components:
    - comp1
    - comp2
    - comp3
    - comp4
    """

    df = base_df.copy()
    maint = maint.copy()

    df = _to_datetime(df)
    maint = _to_datetime(maint)

    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    components = ["comp1", "comp2", "comp3", "comp4"]

    for comp in components:
        df[f"hours_since_maint_{comp}"] = np.nan

    # Efficient and deterministic enough for assignment scale
    for machine_id in df["machineID"].unique():
        machine_mask = df["machineID"] == machine_id
        machine_times = df.loc[machine_mask, "datetime"]

        machine_maint = maint[maint["machineID"] == machine_id].copy()

        for comp in components:
            comp_events = (
                machine_maint[machine_maint["comp"] == comp]
                .sort_values("datetime")["datetime"]
                .tolist()
            )

            last_maint_times = []
            event_idx = 0
            last_time = pd.NaT

            # Add 24-hour buffer: only use maintenance from 24+ hours ago
            buffer_hours = 24
            buffer_td = pd.Timedelta(hours=buffer_hours)

            for current_time in machine_times:
                cutoff_time = current_time - buffer_td  # Only look back 24+ hours
                while event_idx < len(comp_events) and comp_events[event_idx] <= cutoff_time:
                    last_time = comp_events[event_idx]
                    event_idx += 1

                last_maint_times.append(last_time)

            if len(last_maint_times) > 0:
                last_series = pd.Series(last_maint_times, index=df.loc[machine_mask].index)

                hours_since = (
                    df.loc[machine_mask, "datetime"] - last_series
                ).dt.total_seconds() / 3600

                df.loc[machine_mask, f"hours_since_maint_{comp}"] = hours_since

    # If no previous maintenance exists, use large value
    maint_cols = [col for col in df.columns if col.startswith("hours_since_maint_")]
    df[maint_cols] = df[maint_cols].fillna(9999)

    # CRITICAL: Shift maintenance features by 72 hours to prevent leakage
    # Maintenance data likely records repairs AFTER failures
    # At time t, use maintenance from 96+ hours ago, not from the current or recent window
    for maint_col in maint_cols:
        df[maint_col] = df.groupby("machineID")[maint_col].shift(72).fillna(9999)

    return df


def add_machine_metadata(base_df: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """
    Add machine age and model metadata.
    Model is one-hot encoded.
    """

    df = base_df.copy()
    machines = machines.copy()

    df = df.merge(machines, on="machineID", how="left")

    if "model" in df.columns:
        df = pd.get_dummies(df, columns=["model"], prefix="model", dtype=int)

    return df


def add_failure_labels(base_df: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """
    Label definition:
    failure_24h = 1 if any failure occurs in (t, t+24h], else 0.

    Also stores future_failure_component for analysis/output, but this column
    must NOT be used as a model feature.
    """

    df = base_df.copy()
    failures = failures.copy()

    df = _to_datetime(df)
    failures = _to_datetime(failures)

    df["failure_24h"] = 0
    df["future_failure_component"] = "none"

    failures = failures.sort_values(["machineID", "datetime"])

    for _, failure_row in failures.iterrows():
        machine_id = failure_row["machineID"]
        failure_time = failure_row["datetime"]
        component = failure_row["failure"]

        # Label = 1 if failure occurs in (t, t+24h]
        # Rearranged for rows t:
        # t >= failure_time - 24h and t < failure_time
        start_time = failure_time - pd.Timedelta(hours=24)

        mask = (
            (df["machineID"] == machine_id)
            & (df["datetime"] >= start_time)
            & (df["datetime"] < failure_time)
        )

        df.loc[mask, "failure_24h"] = 1
        df.loc[mask, "future_failure_component"] = component

    return df


def clean_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup:
    - sort deterministically
    - fill numeric missing values
    - remove duplicate columns if any
    - prevent merge artifacts from entering silently
    """

    df = df.copy()

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def build_features(data: dict, save: bool = True) -> pd.DataFrame:
    """
    Full deterministic feature pipeline.
    """

    telemetry = data["telemetry"].copy()
    errors = data["errors"].copy()
    maint = data["maint"].copy()
    failures = data["failures"].copy()
    machines = data["machines"].copy()

    telemetry = _to_datetime(telemetry)
    errors = _to_datetime(errors)
    maint = _to_datetime(maint)
    failures = _to_datetime(failures)

    features = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    features = add_rolling_telemetry_features(features)
    features = add_error_features(features, errors)
    features = add_maintenance_features(features, maint)
    features = add_machine_metadata(features, machines)
    features = add_failure_labels(features, failures)
    features = clean_feature_table(features)

    if save:
        output_path = ARTIFACTS_DIR / "features.parquet"
        features.to_parquet(output_path, index=False)
        print(f"Saved feature table: {output_path.resolve()}")

    return features


def build_feature_table(data: dict, save: bool = True) -> pd.DataFrame:
    """
    Backward-compatible wrapper for pipeline.py.
    Existing pipeline imports build_feature_table, so this calls build_features.
    """
    return build_features(data=data, save=save)