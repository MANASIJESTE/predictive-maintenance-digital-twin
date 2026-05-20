"""
DIAGNOSTIC SCRIPT: Verify data leakage sources

Run this to understand the timing relationships between errors, maintenance, and failures.
This will give you concrete evidence for your interview about what's leaking.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def check_error_failure_timing():
    """Check if errors occur before or at the same time as failures."""
    
    DATA_RAW_DIR = Path("data/raw")
    
    errors = pd.read_csv(DATA_RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    failures = pd.read_csv(DATA_RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    
    print("=" * 70)
    print("ERROR-TO-FAILURE TIMING ANALYSIS")
    print("=" * 70)
    
    time_diff_hours = []
    errors_before_failure = 0
    errors_same_time = 0
    errors_after_failure = 0
    
    for idx, failure_row in failures.iterrows():
        machine_id = failure_row["machineID"]
        failure_time = failure_row["datetime"]
        failure_component = failure_row["failure"]
        
        machine_errors = errors[errors["machineID"] == machine_id]
        
        # Find closest error before failure
        errors_before = machine_errors[machine_errors["datetime"] < failure_time]
        errors_same = machine_errors[machine_errors["datetime"] == failure_time]
        errors_after = machine_errors[machine_errors["datetime"] > failure_time]
        
        if len(errors_same) > 0:
            errors_same_time += 1
            time_diff_hours.append(0)
        elif len(errors_before) > 0:
            closest_error_time = errors_before["datetime"].max()
            hours_diff = (failure_time - closest_error_time).total_seconds() / 3600
            time_diff_hours.append(hours_diff)
            if hours_diff <= 24:
                errors_before_failure += 1
        elif len(errors_after) > 0:
            closest_error_time = errors_after["datetime"].min()
            hours_diff = (closest_error_time - failure_time).total_seconds() / 3600
            time_diff_hours.append(-hours_diff)
            errors_after_failure += 1
    
    print(f"\n📊 RESULTS (out of {len(failures)} failures):\n")
    print(f"  Errors AT SAME TIME as failure:  {errors_same_time:3d} ({errors_same_time/len(failures)*100:5.1f}%)")
    print(f"  Errors BEFORE failure (≤24h):    {errors_before_failure:3d} ({errors_before_failure/len(failures)*100:5.1f}%)")  
    print(f"  Errors AFTER failure:            {errors_after_failure:3d} ({errors_after_failure/len(failures)*100:5.1f}%)")
    
    if time_diff_hours:
        time_diff_arr = np.array(time_diff_hours)
        print(f"\n⏱️  Time between error and failure (hours):")
        print(f"  Mean:    {time_diff_arr.mean():7.1f}h")
        print(f"  Median:  {np.median(time_diff_arr):7.1f}h")
        print(f"  Min:     {time_diff_arr.min():7.1f}h")
        print(f"  Max:     {time_diff_arr.max():7.1f}h")
    
    print("\n⚠️  INTERPRETATION:")
    if errors_same_time > len(failures) * 0.1:
        print(f"  ❌ LEAKAGE LIKELY: {errors_same_time/len(failures)*100:.1f}% of failures have errors at same time!")
        print(f"     → Errors are probably logged AT the failure event, not before")
        print(f"     → Fix: Use errors from t-48h to t-24h, never from prediction window")
    elif errors_before_failure > len(failures) * 0.5:
        print(f"  ⚠️  CAUTION: {errors_before_failure/len(failures)*100:.1f}% of failures preceded by errors")
        print(f"     → Could be legitimate pre-failure signals OR leakage")
        print(f"     → Need to check if errors are strong predictors or just co-occurrence")
    else:
        print(f"  ✓ Errors don't strongly correlate in time with failures")
        print(f"    → Sensor features may be more important than error counts")


def check_maintenance_failure_timing():
    """Check if maintenance occurs before or after failures."""
    
    DATA_RAW_DIR = Path("data/raw")
    
    maint = pd.read_csv(DATA_RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    failures = pd.read_csv(DATA_RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    
    print("\n" + "=" * 70)
    print("MAINTENANCE-TO-FAILURE TIMING ANALYSIS")
    print("=" * 70)
    
    maint_before_failure = 0
    maint_after_failure = 0
    maint_same_component = 0
    
    for idx, failure_row in failures.iterrows():
        machine_id = failure_row["machineID"]
        failure_time = failure_row["datetime"]
        failure_component = failure_row["failure"]
        
        machine_maint = maint[maint["machineID"] == machine_id]
        
        # Check if maintenance happened for the same component near the failure
        comp_maint = machine_maint[machine_maint["comp"] == failure_component]
        
        maint_before = comp_maint[comp_maint["datetime"] < failure_time]
        maint_after = comp_maint[comp_maint["datetime"] >= failure_time]
        
        if len(maint_after) > 0:
            maint_after_failure += 1
            if len(maint_before) > 0:
                maint_same_component += 1
        elif len(maint_before) > 0:
            maint_before_failure += 1
    
    print(f"\n📊 RESULTS (out of {len(failures)} failures):\n")
    print(f"  Maintenance BEFORE failure:      {maint_before_failure:3d} ({maint_before_failure/len(failures)*100:5.1f}%)")
    print(f"  Maintenance AFTER failure:       {maint_after_failure:3d} ({maint_after_failure/len(failures)*100:5.1f}%)")
    print(f"  Same component both times:       {maint_same_component:3d} ({maint_same_component/len(failures)*100:5.1f}%)")
    
    print("\n⚠️  INTERPRETATION:")
    if maint_after_failure > len(failures) * 0.3:
        print(f"  ❌ LEAKAGE LIKELY: {maint_after_failure/len(failures)*100:.1f}% of failures followed by maintenance!")
        print(f"     → Maintenance data may include post-failure repairs")
        print(f"     → Fix: Only use maintenance from 48+ hours before prediction window")
    else:
        print(f"  ✓ Most maintenance happens before failures (preventive or unrelated)")
        print(f"    → But still add temporal buffer to be safe")


def check_feature_correlation_strength():
    """Check which feature groups are too predictive."""
    
    ARTIFACTS_DIR = Path("artifacts")
    
    if not (ARTIFACTS_DIR / "features.parquet").exists():
        print("\n⚠️  Features file not found. Run: python -m src.pipeline first")
        return
    
    features_df = pd.read_parquet(ARTIFACTS_DIR / "features.parquet")
    
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    if "failure_24h" in features_df.columns:
        # Separate features by type
        error_features = [col for col in features_df.columns if "error" in col.lower() and "count" in col]
        maint_features = [col for col in features_df.columns if "maint" in col.lower()]
        sensor_features = [col for col in features_df.columns if any(s in col.lower() for s in ["volt", "rotate", "pressure", "vibration"])]
        
        print(f"\n📊 FEATURE GROUPS:\n")
        print(f"  Error features:   {len(error_features)}")
        print(f"  Maint features:   {len(maint_features)}")
        print(f"  Sensor features:  {len(sensor_features)}")
        
        correlations = features_df[error_features + maint_features + sensor_features + ["failure_24h"]].corr()["failure_24h"]
        
        print(f"\n📈 CORRELATION WITH failure_24h:\n")
        
        if error_features:
            error_corrs = correlations[error_features].abs().sort_values(ascending=False)
            print(f"  Error features (top 3):")
            for col, corr in error_corrs.head(3).items():
                print(f"    {col}: {corr:.4f}")
                if corr > 0.9:
                    print(f"      ⚠️  TOO HIGH - suggests leakage!")
        
        if maint_features:
            maint_corrs = correlations[maint_features].abs().sort_values(ascending=False)
            print(f"\n  Maint features (top 3):")
            for col, corr in maint_corrs.head(3).items():
                print(f"    {col}: {corr:.4f}")
        
        if sensor_features:
            sensor_corrs = correlations[sensor_features].abs().sort_values(ascending=False)
            print(f"\n  Sensor features (top 3):")
            for col, corr in sensor_corrs.head(3).items():
                print(f"    {col}: {corr:.4f}")


if __name__ == "__main__":
    print("\n🔍 PREDICTIVE MAINTENANCE - LEAKAGE INVESTIGATION\n")
    
    check_error_failure_timing()
    check_maintenance_failure_timing()
    check_feature_correlation_strength()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR INTERVIEW")
    print("=" * 70)
    print("""
1. Use findings above to explain leakage to interviewer
2. Focus on: "I found ROC-AUC 0.999961 was unrealistic, diagnosed it as leakage"
3. Show the timing analysis (errors at same time as failures?)
4. Explain the fix: implement temporal separation with buffer periods
5. Acknowledge: "In production, we'd never see this performance"
    """)
