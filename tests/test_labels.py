import pandas as pd

from src.labels import add_failure_label


def test_add_failure_label_open_left_closed_right_interval():
    base = pd.DataFrame(
        {
            "machineID": [1, 1, 1],
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-02 00:00:00"]
            ),
        }
    )
    failures = pd.DataFrame(
        {
            "machineID": [1],
            "datetime": pd.to_datetime(["2020-01-02 00:00:00"]),
            "failure": ["comp1"],
        }
    )

    labeled = add_failure_label(base, failures, horizon_hours=24)

    # 2020-01-01 00:00 includes failure exactly at t+24h.
    assert labeled.loc[0, "failure_24h"] == 1

    # 2020-01-01 01:00 also includes the future failure within 24h.
    assert labeled.loc[1, "failure_24h"] == 1

    # At exact failure timestamp, interval is (t, t+24h], so the same failure is not counted.
    assert labeled.loc[2, "failure_24h"] == 0
