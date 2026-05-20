from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENT_RULES = {
    "comp1": ["volt", "error1", "days_since_comp1_maint"],
    "comp2": ["rotate", "error2", "days_since_comp2_maint"],
    "comp3": ["pressure", "error3", "days_since_comp3_maint"],
    "comp4": ["vibration", "error4", "error5", "days_since_comp4_maint"],
}


def health_state_from_risk(risk: float) -> str:
    if risk < 0.25:
        return "healthy"
    if risk < 0.50:
        return "watch"
    if risk < 0.75:
        return "degraded"
    return "critical"


def prescription_from_state(state: str) -> str:
    return {
        "healthy": "continue",
        "watch": "monitor",
        "degraded": "schedule_maintenance",
        "critical": "urgent_maintenance",
    }[state]


def confidence_from_risk(risk: float) -> str:
    distance_from_middle = abs(risk - 0.5)
    if distance_from_middle < 0.15:
        return "low"
    if distance_from_middle < 0.30:
        return "medium"
    return "high"


def likely_component_from_evidence(row: pd.Series, train_reference: pd.DataFrame | None = None) -> str:
    """Rule-based component mapping from interpretable evidence.

    The assignment allows a rule-based mapping. Here, each component receives a score
    when its related signals are unusually high or maintenance is old.
    """
    scores = {component: 0.0 for component in COMPONENT_RULES}

    for component, keywords in COMPONENT_RULES.items():
        for feature, value in row.items():
            feature_name = str(feature)
            if any(keyword in feature_name for keyword in keywords):
                if feature_name.startswith("days_since"):
                    scores[component] += min(float(value) / 365.0, 3.0)
                elif "count" in feature_name:
                    scores[component] += float(value)
                elif train_reference is not None and feature_name in train_reference.columns:
                    mean = train_reference[feature_name].mean()
                    std = train_reference[feature_name].std() or 1.0
                    scores[component] += abs(float(value) - mean) / std

    best_component = max(scores, key=scores.get)
    return best_component if scores[best_component] > 0 else "unknown"


def evidence_items(row: pd.Series, train_reference: pd.DataFrame, feature_columns: list[str], max_items: int = 3) -> list[str]:
    evidence = []
    scored = []
    for col in feature_columns:
        if col not in train_reference.columns or col not in row:
            continue
        std = train_reference[col].std()
        if std == 0 or pd.isna(std):
            continue
        z = abs(float(row[col]) - train_reference[col].mean()) / std
        scored.append((z, col, row[col]))

    for z, col, value in sorted(scored, reverse=True)[:max_items]:
        readable = col.replace("_", " ")
        evidence.append(f"{readable} is unusual compared with training baseline")

    if not evidence:
        evidence.append("No strong abnormal feature found; risk is driven by combined model signal")
    return evidence[:max_items]


def make_digital_twin_response(
    machine_id: int,
    timestamp: str,
    risk: float,
    row: pd.Series,
    train_reference: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    state = health_state_from_risk(risk)
    return {
        "machineID": int(machine_id),
        "timestamp": str(timestamp),
        "failure_risk_24h": round(float(risk), 4),
        "health_state": state,
        "likely_component": likely_component_from_evidence(row, train_reference),
        "confidence": confidence_from_risk(float(risk)),
        "main_evidence": evidence_items(row, train_reference, feature_columns, max_items=3),
        "prescription": prescription_from_state(state),
    }
