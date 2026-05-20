import pandas as pd
from src.config import LABEL_HORIZON_HOURS


def add_failure_label(base: pd.DataFrame, failures: pd.DataFrame, horizon_hours: int = LABEL_HORIZON_HOURS) -> pd.DataFrame:
    """Label = 1 when any failure occurs in (t, t + horizon].

    This is intentionally time-aware and avoids looking backward from failures.
    The interval is open on the left and closed on the right.
    """
    result = base.copy()
    result["failure_24h"] = 0
    result["future_failure_component"] = "none"

    if failures.empty:
        return result

    failures_small = failures[["machineID", "datetime", "failure"]].copy()
    failures_small = failures_small.rename(columns={"datetime": "failure_datetime"})

    labeled_chunks = []
    for machine_id, machine_rows in result.groupby("machineID", sort=True):
        machine_rows = machine_rows.sort_values("datetime").copy()
        machine_failures = failures_small[failures_small["machineID"] == machine_id].sort_values("failure_datetime")

        if machine_failures.empty:
            labeled_chunks.append(machine_rows)
            continue

        failure_times = machine_failures["failure_datetime"].to_numpy()
        failure_components = machine_failures["failure"].to_numpy()

        labels = []
        components = []
        for timestamp in machine_rows["datetime"]:
            upper = timestamp + pd.Timedelta(hours=horizon_hours)
            mask = (machine_failures["failure_datetime"] > timestamp) & (machine_failures["failure_datetime"] <= upper)
            if mask.any():
                first_match = machine_failures.loc[mask].iloc[0]
                labels.append(1)
                components.append(first_match["failure"])
            else:
                labels.append(0)
                components.append("none")

        machine_rows["failure_24h"] = labels
        machine_rows["future_failure_component"] = components
        labeled_chunks.append(machine_rows)

    return pd.concat(labeled_chunks, ignore_index=True)
