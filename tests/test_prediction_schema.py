import pandas as pd

from src.digital_twin import make_digital_twin_response


def test_digital_twin_response_schema():
    row = pd.Series(
        {
            "machineID": 1,
            "volt_mean_24h": 200.0,
            "error1_count_24h": 2.0,
            "days_since_comp1_maint": 100.0,
        }
    )
    train_reference = pd.DataFrame(
        {
            "volt_mean_24h": [160.0, 161.0, 159.0],
            "error1_count_24h": [0.0, 0.0, 1.0],
            "days_since_comp1_maint": [10.0, 20.0, 30.0],
        }
    )
    feature_columns = list(train_reference.columns)

    response = make_digital_twin_response(
        machine_id=1,
        timestamp="2020-01-01 00:00:00",
        risk=0.82,
        row=row,
        train_reference=train_reference,
        feature_columns=feature_columns,
    )

    expected_keys = {
        "machineID",
        "timestamp",
        "failure_risk_24h",
        "health_state",
        "likely_component",
        "confidence",
        "main_evidence",
        "prescription",
    }
    assert set(response.keys()) == expected_keys
    assert response["health_state"] in {"healthy", "watch", "degraded", "critical"}
    assert response["prescription"] in {
        "continue",
        "monitor",
        "inspect",
        "schedule_maintenance",
        "urgent_maintenance",
    }
    assert len(response["main_evidence"]) <= 3
