from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src.ingest import load_raw_data
from src.validate import validate_raw_data


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_eda():
    data = load_raw_data()
    validate_raw_data(data)

    telemetry = data["telemetry"].copy()
    errors = data["errors"].copy()
    maintenance = data["maint"].copy()
    failures = data["failures"].copy()
    machines = data["machines"].copy()

    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    maintenance["datetime"] = pd.to_datetime(maintenance["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    summary = {}

    summary["telemetry_rows"] = len(telemetry)
    summary["errors_rows"] = len(errors)
    summary["maintenance_rows"] = len(maintenance)
    summary["failures_rows"] = len(failures)
    summary["machines_rows"] = len(machines)

    summary["n_machines"] = machines["machineID"].nunique()
    summary["telemetry_start"] = str(telemetry["datetime"].min())
    summary["telemetry_end"] = str(telemetry["datetime"].max())
    summary["n_failure_events"] = len(failures)
    summary["n_error_events"] = len(errors)
    summary["n_maintenance_events"] = len(maintenance)

    # Missing values
    missing_report = {
        "telemetry": telemetry.isna().sum().to_dict(),
        "errors": errors.isna().sum().to_dict(),
        "maintenance": maintenance.isna().sum().to_dict(),
        "failures": failures.isna().sum().to_dict(),
        "machines": machines.isna().sum().to_dict(),
    }

    # Save summary
    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    summary_df.to_csv(REPORTS_DIR / "eda_summary.csv", index=False)

    missing_rows = []

    for table_name, values in missing_report.items():
        for column, missing_count in values.items():
            missing_rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "missing_count": missing_count,
                }
            )

    pd.DataFrame(missing_rows).to_csv(
        REPORTS_DIR / "eda_missing_values.csv",
        index=False,
    )

    # Failure distribution
    failure_counts = failures["failure"].value_counts().sort_index()
    failure_counts.to_csv(REPORTS_DIR / "eda_failure_counts.csv")

    plt.figure(figsize=(7, 4))
    failure_counts.plot(kind="bar")
    plt.title("Failure count by component")
    plt.xlabel("Component")
    plt.ylabel("Failure count")
    save_plot(REPORTS_DIR / "eda_failure_count_by_component.png")

    # Error distribution
    error_counts = errors["errorID"].value_counts().sort_index()
    error_counts.to_csv(REPORTS_DIR / "eda_error_counts.csv")

    plt.figure(figsize=(7, 4))
    error_counts.plot(kind="bar")
    plt.title("Error count by error type")
    plt.xlabel("Error type")
    plt.ylabel("Count")
    save_plot(REPORTS_DIR / "eda_error_count_by_type.png")

    # Maintenance distribution
    maint_counts = maintenance["comp"].value_counts().sort_index()
    maint_counts.to_csv(REPORTS_DIR / "eda_maintenance_counts.csv")

    plt.figure(figsize=(7, 4))
    maint_counts.plot(kind="bar")
    plt.title("Maintenance count by component")
    plt.xlabel("Component")
    plt.ylabel("Maintenance count")
    save_plot(REPORTS_DIR / "eda_maintenance_count_by_component.png")

    # Machine model distribution
    model_counts = machines["model"].value_counts().sort_index()
    model_counts.to_csv(REPORTS_DIR / "eda_machine_model_counts.csv")

    plt.figure(figsize=(7, 4))
    model_counts.plot(kind="bar")
    plt.title("Machine count by model")
    plt.xlabel("Machine model")
    plt.ylabel("Machine count")
    save_plot(REPORTS_DIR / "eda_machine_count_by_model.png")

    # Machine age distribution
    plt.figure(figsize=(7, 4))
    machines["age"].plot(kind="hist", bins=20)
    plt.title("Machine age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot(REPORTS_DIR / "eda_machine_age_distribution.png")

    # Telemetry summary
    telemetry_cols = ["volt", "rotate", "pressure", "vibration"]
    telemetry_summary = telemetry[telemetry_cols].describe().T
    telemetry_summary.to_csv(REPORTS_DIR / "eda_telemetry_summary.csv")

    # Telemetry distributions
    for col in telemetry_cols:
        plt.figure(figsize=(7, 4))
        telemetry[col].plot(kind="hist", bins=50)
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(REPORTS_DIR / f"eda_{col}_distribution.png")

    # Failures over time
    failures_by_month = (
        failures
        .set_index("datetime")
        .resample("ME")
        .size()
    )

    failures_by_month.to_csv(REPORTS_DIR / "eda_failures_by_month.csv")

    plt.figure(figsize=(9, 4))
    failures_by_month.plot()
    plt.title("Failures over time")
    plt.xlabel("Month")
    plt.ylabel("Failure count")
    save_plot(REPORTS_DIR / "eda_failures_over_time.png")

    # Errors over time
    errors_by_month = (
        errors
        .set_index("datetime")
        .resample("ME")
        .size()
    )

    errors_by_month.to_csv(REPORTS_DIR / "eda_errors_by_month.csv")

    plt.figure(figsize=(9, 4))
    errors_by_month.plot()
    plt.title("Errors over time")
    plt.xlabel("Month")
    plt.ylabel("Error count")
    save_plot(REPORTS_DIR / "eda_errors_over_time.png")

    # Feature Correlation Analysis (if features exist)
    features_path = Path("artifacts/features.parquet")
    if features_path.exists():
        print("\n" + "="*60)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*60)
        
        features_df = pd.read_parquet(features_path)
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_cols = ["failure_24h", "machineID", "datetime", "label", "failure"]
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        if "failure_24h" in features_df.columns and len(feature_cols) > 0:
            # Calculate correlations
            correlations = features_df[feature_cols + ["failure_24h"]].corr()["failure_24h"].drop("failure_24h")
            correlations_abs = correlations.abs().sort_values(ascending=False)
            
            print(f"\n📊 Top 20 Most Correlated Features with Target (failure_24h):\n")
            for i, (feat, corr) in enumerate(correlations_abs.head(20).items(), 1):
                actual_corr = correlations[feat]
                print(f"{i:2d}. {feat:35s} | {actual_corr:+.6f}")
            
            # Save to CSV
            corr_report = pd.DataFrame({
                "feature": correlations_abs.index,
                "correlation": correlations[correlations_abs.index].values,
                "abs_correlation": correlations_abs.values
            })
            corr_report.to_csv(REPORTS_DIR / "feature_correlation_ranking.csv", index=False)
            
            # Feature Groups Analysis
            sensor_features = [col for col in feature_cols if any(s in col.lower() for s in ["volt", "rotate", "pressure", "vibration"])]
            error_features = [col for col in feature_cols if "error" in col.lower()]
            maint_features = [col for col in feature_cols if "maint" in col.lower()]
            model_features = [col for col in feature_cols if "model" in col.lower()]
            
            print(f"\n📈 Feature Groups by Average Correlation:\n")
            groups = {
                "Sensor Features": sensor_features,
                "Error Count Features": error_features,
                "Maintenance Features": maint_features,
                "Model/Machine Features": model_features,
            }
            
            for group_name, cols in groups.items():
                if cols:
                    avg_corr = correlations[cols].abs().mean()
                    max_corr = correlations[cols].abs().max()
                    print(f"  {group_name:30s} | Avg: {avg_corr:.4f} | Max: {max_corr:.4f} | Count: {len(cols)}")
            
            # Heatmap of top 15 features
            top_15 = correlations_abs.head(15).index.tolist()
            plt.figure(figsize=(8, 6))
            corr_data = features_df[top_15+ ["failure_24h"]].corr()
            sns.heatmap(corr_data.iloc[:-1, -1:], annot=True, fmt=".4f", cmap="RdBu_r", center=0,
                       cbar_kws={"label": "Correlation"}, linewidths=0.5)
            plt.title("Top 15 Features - Correlation with Target", fontsize=12, fontweight="bold")
            save_plot(REPORTS_DIR / "feature_correlation_heatmap.png")
            
            # Bar chart of top 15
            plt.figure(figsize=(10, 6))
            top_15_corr = correlations_abs.head(15)
            colors = ["#C44E52" if correlations[feat] < 0 else "#55A868" for feat in top_15_corr.index]
            plt.barh(range(len(top_15_corr)), top_15_corr.values, color=colors, alpha=0.8, edgecolor="white")
            plt.yticks(range(len(top_15_corr)), top_15_corr.index, fontsize=9)
            plt.xlabel("Absolute Correlation with Failure Target", fontsize=10)
            plt.title("Top 15 Features by Correlation Strength", fontsize=12, fontweight="bold")
            plt.gca().invert_yaxis()
            for i, (feat, val) in enumerate(top_15_corr.items()):
                plt.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8)
            save_plot(REPORTS_DIR / "feature_correlation_ranking_chart.png")
            
            print("\n✓ Feature correlation reports saved to reports/")
        else:
            print("⚠️ Features not ready. Run pipeline first.")

    print("\nEDA complete. Reports saved in:", REPORTS_DIR.resolve())


if __name__ == "__main__":
    run_eda()