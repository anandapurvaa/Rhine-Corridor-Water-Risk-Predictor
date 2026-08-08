from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import joblib
import pandas as pd

from modeling.data_loader import load_bigquery_table, write_bigquery_table
from mlops.gcs_utils import download_blob, download_json
from modeling.schemas import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS_LEAN,
    PRODUCTION_FEATURE_COLUMNS,
    ROBUSTNESS_REQUIRED_COLUMNS,
    TRAIN_TABLE_NAME,
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"

PREDICTIONS_CSV = OUTPUT_DIR / "gauge_24h_production_predictions.csv"
PREDICTIONS_SUMMARY_JSON = OUTPUT_DIR / "gauge_24h_production_predictions_summary.json"

# Base name; we will append _validation or _test
PREDICTIONS_TABLE_BASE = "gauge_24h_production_predictions"

WEATHER_REQUIRED_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
]

# Which split to predict on: "validation" or "test"
PRED_SPLIT_ENV = os.getenv("GAUGE24H_PRED_SPLIT", "test").strip().lower()
if PRED_SPLIT_ENV not in {"validation", "test"}:
    raise ValueError("GAUGE24H_PRED_SPLIT must be 'validation' or 'test'")

HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))


def make_run_id(model_version: str, split_name: str) -> str:
    seed = f"{model_version}|{split_name}|{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def load_production_artifacts() -> tuple[dict, object, str]:
    registry_df = load_bigquery_table(
        "model_registry",
        dataset="mlops",
        columns=[
            "model_version",
            "status",
            "gcs_path",
            "promoted_at_utc",
        ],
        where_sql="status = 'prod'",
        order_by="promoted_at_utc DESC",
        allow_missing_columns=False,
    )

    if registry_df.empty:
        raise RuntimeError(
            "No production model found in mlops.model_registry."
        )

    registry_row = registry_df.iloc[0]
    model_version = str(registry_row["model_version"])
    model_gcs_path = str(registry_row["gcs_path"])

    if not model_gcs_path.startswith("gs://"):
        raise ValueError(
            f"Invalid production model GCS path: {model_gcs_path}"
        )

    bucket_name = model_gcs_path[5:].split("/", 1)[0]
    summary_gcs_path = (
        f"gs://{bucket_name}/artifacts/"
        f"{model_version}/training_summary.json"
    )

    model_path = OUTPUT_DIR / f"model_{model_version}.joblib"
    download_blob(model_gcs_path, model_path)

    training_summary = download_json(summary_gcs_path)

    summary_version = training_summary.get("model_version")
    if summary_version != model_version:
        raise RuntimeError(
            "Model/summary version mismatch: "
            f"registry={model_version}, summary={summary_version}"
        )

    model = joblib.load(model_path)

    print(f"Loaded production model: {model_version}")
    print(f"Model source: {model_gcs_path}")
    print(f"Summary source: {summary_gcs_path}")

    return training_summary, model, model_version


def resolve_feature_columns(training_summary: dict, df: pd.DataFrame) -> list[str]:
    trained_features = training_summary.get("feature_columns_used") or training_summary.get("feature_columns") or []
    if trained_features:
        return [c for c in trained_features if c in df.columns]
    return [c for c in PRODUCTION_FEATURE_COLUMNS if c in df.columns]


def prepare_inference_frame(horizon_hours: int, training_summary: dict) -> tuple[pd.DataFrame, str, list[str]]:
    target_column = f"target_value_t_plus_{horizon_hours}h"
    requested = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS
            + [DEFAULT_TARGET_COLUMN, target_column, "target_value", "timestamp_utc", "split_name"]
        )
    )

    df = load_bigquery_table(
        TRAIN_TABLE_NAME,
        columns=requested,
        order_by="timestamp_utc, station_name",
        allow_missing_columns=True,
    )

    missing_required = [c for c in ROBUSTNESS_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required inference base columns: {missing_required}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    if "split_name" not in df.columns:
        raise ValueError("Inference table must include split_name column.")
    df["split_name"] = df["split_name"].astype("string").str.lower()
    df = df[df["split_name"] == PRED_SPLIT_ENV].copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    numeric_cols = list(dict.fromkeys(NUMERIC_COLUMNS_LEAN + [DEFAULT_TARGET_COLUMN, target_column, "target_value"]))
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require only weather columns to be non-null
    weather_ok = df[WEATHER_REQUIRED_COLUMNS].notna().all(axis=1)
    df = df.loc[weather_ok].copy()

    latest_rows = (
        df.groupby(["split_name", "station_name"], dropna=False)["timestamp_utc"]
        .max()
        .rename("latest_timestamp_utc")
        .reset_index()
    )

    latest_df = df.merge(
        latest_rows,
        on=["split_name", "station_name"],
        how="inner",
        validate="many_to_one",
    )
    latest_df = latest_df[latest_df["timestamp_utc"] == latest_df["latest_timestamp_utc"]].copy()
    latest_df = latest_df.sort_values(["split_name", "station_name", "timestamp_utc"]).reset_index(drop=True)

    feature_columns = resolve_feature_columns(training_summary, latest_df)
    missing_trained = [c for c in (training_summary.get("feature_columns_used") or []) if c not in latest_df.columns]
    if missing_trained:
        raise ValueError(f"Inference data is missing trained feature columns: {missing_trained}")
    if not feature_columns:
        raise ValueError("No usable feature columns available for inference.")

    latest_df["forecast_timestamp_utc"] = latest_df["timestamp_utc"] + pd.to_timedelta(horizon_hours, unit="h")
    latest_df["prediction_ready_utc"] = pd.Timestamp.now(tz="UTC")
    return latest_df, target_column, feature_columns


def validate_inference_frame(df: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [c for c in ["timestamp_utc", "target_value", "split_name"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")

    if df.empty:
        raise ValueError("Inference frame is empty")

    if df["station_name"].isna().all():
        raise ValueError("All station_name values are null")

    duplicates = df.duplicated(subset=["split_name", "station_name"], keep=False)
    if duplicates.any():
        dupes = df.loc[duplicates, ["split_name", "station_name", "timestamp_utc"]].sort_values(["split_name", "station_name"])
        raise ValueError(f"Expected one latest row per split/station, found duplicates:\n{dupes.head(20)}")

    if not feature_columns:
        raise ValueError("No feature columns resolved for inference.")


def score_frame(
    df: pd.DataFrame,
    model,
    model_version: str,
    target_mode: str,
    horizon_hours: int,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = df.copy()
    X = result[feature_columns].copy()

    pred_raw = model.predict(X)

    if target_mode == "delta":
        y_now = pd.to_numeric(
            result["target_value"],
            errors="coerce",
        ).to_numpy()

        result["prediction"] = pred_raw + y_now

    elif target_mode == "level":
        result["prediction"] = pred_raw

    else:
        raise ValueError(
            f"Unsupported target_mode={target_mode!r}"
        )

    pipeline_run_id = os.getenv(
        "MLOPS_RUN_ID"
    )

    if not pipeline_run_id:
        pipeline_run_id = make_run_id(
            model_version,
            PRED_SPLIT_ENV,
        )

    result["model_version"] = model_version
    result["run_id"] = pipeline_run_id
    result["target_column"] = target_column
    result["target_mode"] = target_mode
    result["prediction_horizon_hours"] = (
        horizon_hours
    )

    if target_column in result.columns:
        result["actual_if_available"] = (
            result[target_column]
        )
        result["actual_available_now"] = (
            result[target_column].notna()
        )
    else:
        result["actual_if_available"] = pd.NA
        result["actual_available_now"] = False

    output_cols = [
        "run_id",
        "model_version",
        "split_name",
        "station_name",
        "timeseries_name",
        "unit",
        "source",
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction_ready_utc",
        "prediction_horizon_hours",
        "target_mode",
        "prediction",
        "actual_if_available",
        "actual_available_now",
        "target_column",
    ] + feature_columns

    output_cols = [
        column
        for column in output_cols
        if column in result.columns
    ]

    output_cols = list(
        dict.fromkeys(output_cols)
    )

    return (
        result[output_cols]
        .sort_values(
            [
                "split_name",
                "station_name",
                "timestamp_utc",
            ]
        )
        .reset_index(drop=True)
    )

def build_summary(pred_df: pd.DataFrame, training_summary: dict, feature_columns: list[str]) -> dict:
    return {
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": pred_df["run_id"].iloc[0] if len(pred_df) else None,
        "model_version": pred_df["model_version"].iloc[0] if len(pred_df) else None,
        "target_column": training_summary.get("target_column"),
        "target_mode": training_summary.get("target_mode"),
        "horizon_hours": training_summary.get("horizon_hours"),
        "training_rows": training_summary.get("rows_trained"),
        "training_train_end_utc": training_summary.get("train_end_utc"),
        "feature_columns_used_for_inference": feature_columns,
        "rows_predicted": int(len(pred_df)),
        "stations_predicted": int(pred_df["station_name"].nunique()) if len(pred_df) else 0,
        "split_present": pred_df["split_name"].iloc[0] if len(pred_df) else None,
        "source_table": TRAIN_TABLE_NAME,
        "prediction_table": f"{PREDICTIONS_TABLE_BASE}_{PRED_SPLIT_ENV}",
        "min_forecast_timestamp_utc": pred_df["forecast_timestamp_utc"].min().isoformat() if len(pred_df) else None,
        "max_forecast_timestamp_utc": pred_df["forecast_timestamp_utc"].max().isoformat() if len(pred_df) else None,
    }


def main():
    training_summary, model, model_version = load_production_artifacts()
    target_mode = training_summary.get("target_mode", "level")
    horizon_hours = HORIZON_HOURS
    inference_df, target_column, feature_columns = prepare_inference_frame(horizon_hours, training_summary)
    validate_inference_frame(inference_df, feature_columns)

    pred_df = score_frame(
        inference_df,
        model=model,
        model_version=model_version,
        target_mode=target_mode,
        horizon_hours=horizon_hours,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    table_name = f"{PREDICTIONS_TABLE_BASE}_{PRED_SPLIT_ENV}"
    write_bigquery_table(pred_df, table_name, dataset="rhein_curated", if_exists="replace")

    summary = build_summary(pred_df, training_summary, feature_columns)
    with open(PREDICTIONS_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    preview_cols = [
        "split_name",
        "station_name",
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction",
        "model_version",
        "run_id",
    ]
    preview_cols = [c for c in preview_cols if c in pred_df.columns]

    print(pred_df[preview_cols].head(30).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {PREDICTIONS_CSV}")
    print(f"Wrote BigQuery table: rhein_curated.{table_name}")
    print(f"Wrote: {PREDICTIONS_SUMMARY_JSON}")
    return summary


if __name__ == "__main__":
    main()