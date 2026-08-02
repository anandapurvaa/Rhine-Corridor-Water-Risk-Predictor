from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import pandas as pd

from modeling.data_loader import load_bigquery_table, write_bigquery_table
from modeling.schemas import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS_LEAN,
    PRODUCTION_FEATURE_COLUMNS,
    TABLE_NAME,
    TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"

PREDICTIONS_CSV = OUTPUT_DIR / "gauge_24h_production_predictions.csv"
PREDICTIONS_SUMMARY_JSON = OUTPUT_DIR / "gauge_24h_production_predictions_summary.json"
PREDICTIONS_TABLE = "gauge_24h_production_predictions"

PREDICTION_HORIZON_HOURS = 24


def make_run_id(model_version: str) -> str:
    seed = f"{model_version}|{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def load_training_summary() -> dict:
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing training summary: {TRAINING_SUMMARY_PATH}. "
            "Run train_gauge_24h_production.py first."
        )
    with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing production model: {MODEL_PATH}. "
            "Run train_gauge_24h_production.py first."
        )
    return joblib.load(MODEL_PATH)


def prepare_inference_frame() -> pd.DataFrame:
    columns = list(dict.fromkeys(PRODUCTION_FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]))
    df = load_bigquery_table(
        TABLE_NAME,
        columns=columns,
        order_by="timestamp_utc, station_name",
    )

    if "timestamp_utc" not in df.columns:
        raise ValueError("Expected timestamp_utc in source table")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS_LEAN + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_rows = (
        df.groupby("station_name", dropna=False)["timestamp_utc"]
        .max()
        .rename("latest_timestamp_utc")
        .reset_index()
    )

    latest_df = df.merge(
        latest_rows,
        on="station_name",
        how="inner",
        validate="many_to_one",
    )
    latest_df = latest_df[latest_df["timestamp_utc"] == latest_df["latest_timestamp_utc"]].copy()
    latest_df = latest_df.sort_values(["station_name", "timestamp_utc"]).reset_index(drop=True)

    latest_df["forecast_timestamp_utc"] = latest_df["timestamp_utc"] + pd.to_timedelta(
        PREDICTION_HORIZON_HOURS, unit="h"
    )
    latest_df["prediction_ready_utc"] = pd.Timestamp.now(tz="UTC")
    return latest_df


def validate_inference_frame(df: pd.DataFrame) -> None:
    missing = [c for c in PRODUCTION_FEATURE_COLUMNS + ["timestamp_utc"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")

    if df.empty:
        raise ValueError("Inference frame is empty")

    if df["station_name"].isna().all():
        raise ValueError("All station_name values are null")

    duplicates = df.duplicated(subset=["station_name"], keep=False)
    if duplicates.any():
        dupes = df.loc[duplicates, ["station_name", "timestamp_utc"]].sort_values("station_name")
        raise ValueError(f"Expected one latest row per station, found duplicates:\n{dupes.head(20)}")


def score_frame(df: pd.DataFrame, model, model_version: str) -> pd.DataFrame:
    result = df.copy()
    X = result[PRODUCTION_FEATURE_COLUMNS].copy()

    result["prediction"] = model.predict(X)
    result["model_version"] = model_version
    result["run_id"] = make_run_id(model_version)
    result["target_column"] = TARGET_COLUMN
    result["prediction_horizon_hours"] = PREDICTION_HORIZON_HOURS

    if TARGET_COLUMN in result.columns:
        result["actual_if_available"] = result[TARGET_COLUMN]
        result["actual_available_now"] = result[TARGET_COLUMN].notna()
    else:
        result["actual_if_available"] = pd.NA
        result["actual_available_now"] = False

    output_cols = [
        "run_id",
        "model_version",
        "station_name",
        "timeseries_name",
        "unit",
        "source",
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction_ready_utc",
        "prediction_horizon_hours",
        "prediction",
        "actual_if_available",
        "actual_available_now",
        "target_column",
    ] + PRODUCTION_FEATURE_COLUMNS

    output_cols = [c for c in output_cols if c in result.columns]
    output_cols = list(dict.fromkeys(output_cols))

    return result[output_cols].sort_values(["station_name", "timestamp_utc"]).reset_index(drop=True)


def build_summary(pred_df: pd.DataFrame, training_summary: dict) -> dict:
    return {
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": pred_df["run_id"].iloc[0] if len(pred_df) else None,
        "model_version": pred_df["model_version"].iloc[0] if len(pred_df) else None,
        "training_rows": training_summary.get("rows_trained"),
        "training_train_end_utc": training_summary.get("train_end_utc"),
        "rows_predicted": int(len(pred_df)),
        "stations_predicted": int(pred_df["station_name"].nunique()) if len(pred_df) else 0,
        "source_table": TABLE_NAME,
        "prediction_table": PREDICTIONS_TABLE,
        "min_forecast_timestamp_utc": pred_df["forecast_timestamp_utc"].min().isoformat() if len(pred_df) else None,
        "max_forecast_timestamp_utc": pred_df["forecast_timestamp_utc"].max().isoformat() if len(pred_df) else None,
    }


def main():
    training_summary = load_training_summary()
    model_version = training_summary["model_version"]

    model = load_model()
    inference_df = prepare_inference_frame()
    validate_inference_frame(inference_df)

    pred_df = score_frame(inference_df, model=model, model_version=model_version)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    write_bigquery_table(pred_df, PREDICTIONS_TABLE, dataset="rhein_curated", if_exists="replace")

    summary = build_summary(pred_df, training_summary)
    with open(PREDICTIONS_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    preview_cols = [
        "station_name",
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction",
        "model_version",
        "run_id",
    ]
    preview_cols = [c for c in preview_cols if c in pred_df.columns]

    print(pred_df[preview_cols].head(20).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {PREDICTIONS_CSV}")
    print(f"Wrote BigQuery table: rhein_curated.{PREDICTIONS_TABLE}")
    print(f"Wrote: {PREDICTIONS_SUMMARY_JSON}")


if __name__ == "__main__":
    main()