from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from modeling.data_loader import load_bigquery_table


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "baseline_gauge_24h_regressor.joblib"
TABLE_NAME = "dataset_splits_gauge_24h"
TARGET_COLUMN = "target_value_t_plus_24h"

CATEGORICAL_COLUMNS = [
    "station_name",
    "timeseries_name",
    "unit",
    "source",
]

IDENTIFIER_COLUMNS = [
    "station_id",
    "dwd_station_id",
    "dwd_station_name",
]

NUMERIC_COLUMNS = [
    "target_value",
    "distance_km",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
    "lag_1",
    "lag_3",
    "lag_6",
    "diff_1",
    "diff_3",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_min_6",
    "rolling_max_6",
    "hour_utc",
    "day_of_week",
    "month",
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_6",
    "temp_lag_12",
    "precip_lag_1",
    "precip_lag_3",
    "precip_lag_6",
    "precip_lag_12",
    "wind_lag_1",
    "wind_lag_3",
    "wind_lag_6",
    "pressure_lag_1",
    "pressure_lag_3",
    "humidity_lag_1",
    "temp_roll_mean_3",
    "temp_roll_mean_6",
    "temp_roll_mean_12",
    "precip_roll_mean_6",
    "precip_roll_mean_12",
    "precip_roll_sum_6",
    "precip_roll_sum_12",
    "precip_roll_sum_24",
    "wind_roll_mean_6",
    "wind_roll_mean_12",
    "pressure_roll_mean_6",
    "pressure_roll_mean_12",
    "humidity_roll_mean_6",
    "humidity_roll_mean_12",
    "pressure_delta_1",
    "pressure_delta_3",
    "temp_change_1_3",
    "precip_accel_12_24",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS + NUMERIC_COLUMNS


def assign_risk_band(abs_delta: float) -> str:
    if pd.isna(abs_delta):
        return "unknown"
    if abs_delta < 10:
        return "normal"
    if abs_delta < 25:
        return "watch"
    if abs_delta < 50:
        return "warning"
    return "critical"


def prepare_dataframe() -> pd.DataFrame:
    df = load_bigquery_table(TABLE_NAME)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    required_cols = FEATURE_COLUMNS + ["timestamp_utc", "station_name", "split_name", "target_value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train baseline first with python -m modeling.train_baseline_gauge_24h"
        )

    model = joblib.load(MODEL_PATH)
    df = prepare_dataframe()

    X = df[FEATURE_COLUMNS].copy()
    preds = model.predict(X)

    scored_df = df[
        [
            "station_id",
            "station_name",
            "timeseries_name",
            "timestamp_utc",
            "split_name",
            "target_value",
        ]
    ].copy()

    scored_df["pred_target_value_t_plus_24h"] = preds
    scored_df["pred_delta_24h"] = scored_df["pred_target_value_t_plus_24h"] - scored_df["target_value"]
    scored_df["pred_abs_delta_24h"] = scored_df["pred_delta_24h"].abs()
    scored_df["risk_band"] = scored_df["pred_abs_delta_24h"].apply(assign_risk_band)

    if TARGET_COLUMN in df.columns:
        y_true = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        scored_df["actual_target_value_t_plus_24h"] = y_true
        scored_df["prediction_error"] = scored_df["actual_target_value_t_plus_24h"] - scored_df["pred_target_value_t_plus_24h"]
        scored_df["abs_error"] = scored_df["prediction_error"].abs()

    latest_station_df = (
        scored_df.sort_values(["station_name", "timestamp_utc"])
        .groupby("station_name", as_index=False)
        .tail(1)
        .sort_values(["risk_band", "pred_abs_delta_24h"], ascending=[True, False])
        .reset_index(drop=True)
    )

    risk_summary_df = (
        latest_station_df.groupby("risk_band", dropna=False)
        .size()
        .reset_index(name="station_count")
        .sort_values("station_count", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "rows_scored": int(len(scored_df)),
        "stations_latest_snapshot": int(len(latest_station_df)),
        "max_pred_abs_delta_24h": float(scored_df["pred_abs_delta_24h"].max()),
        "mean_pred_abs_delta_24h": float(scored_df["pred_abs_delta_24h"].mean()),
    }

    scored_df.to_csv(OUTPUT_DIR / "gauge_24h_scored_rows.csv", index=False)
    latest_station_df.to_csv(OUTPUT_DIR / "gauge_24h_latest_station_snapshot.csv", index=False)
    risk_summary_df.to_csv(OUTPUT_DIR / "gauge_24h_risk_summary.csv", index=False)

    with open(OUTPUT_DIR / "gauge_24h_scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()