from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.cloud import bigquery

from modeling.schemas import (
    PRODUCTION_FEATURE_COLUMNS,
    TRAIN_TABLE_NAME,
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)
from modeling.training_utils import (
    HORIZON_HOURS,
    TARGET_MODE,
    MAX_FEATURE_NULL_FRACTION,
    TRAIN_SPLIT_NAME,
    build_horizon_target_column,
    build_training_target,
    cast_columns,
    filter_sparse_rows,
    build_pipeline,
    load_training_frame,
    validate_training_frame,
    training_null_profile,
    stable_model_version,
    load_bigquery_table
)

from mlops.gcs_utils import upload_blob, upload_json

PROJECT_ID = "rhine-corridor-navigator"
BQ_DATASET = "mlops"
REGISTRY_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.model_registry"
EVAL_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.model_evaluations"

bq_client = bigquery.Client(project=PROJECT_ID)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"

EVAL_SPLIT_NAME = os.getenv("GAUGE24H_EVAL_SPLIT_NAME", "validation").strip().lower()


def evaluate_model(
    model,
    feature_columns: list[str],
    target_column: str,
) -> dict:
    requested = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS
            + [
                DEFAULT_TARGET_COLUMN,
                "target_value",
                "timestamp_utc",
                "split_name",
            ]
        )
    )

    df = load_bigquery_table(
        TRAIN_TABLE_NAME,
        columns=requested,
        where_sql=f"split_name = '{EVAL_SPLIT_NAME}'",
        order_by="timestamp_utc, station_name",
        allow_missing_columns=True,
    )

    if "split_name" not in df.columns:
        raise ValueError(
            "Evaluation table must include split_name column."
        )

    df["split_name"] = (
        df["split_name"]
        .astype("string")
        .str.lower()
    )

    df = df[df["split_name"] == EVAL_SPLIT_NAME].copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp_utc"]).copy()

    df = cast_columns(df)
    df = df.dropna(subset=["target_value"]).copy()

    df = df.sort_values(
        ["timestamp_utc", "station_name"]
    ).reset_index(drop=True)

    df, eval_target_column = build_horizon_target_column(
        df,
        HORIZON_HOURS,
    )

    df[eval_target_column] = pd.to_numeric(
        df[eval_target_column],
        errors="coerce",
    )
    df = df.dropna(subset=[eval_target_column]).copy()

    df = filter_sparse_rows(df, feature_columns)

    X = df[feature_columns].copy()

    y_level = pd.to_numeric(
        df[eval_target_column],
        errors="coerce",
    )

    if TARGET_MODE == "delta":
        y_now = pd.to_numeric(
            df["target_value"],
            errors="coerce",
        )
        y = y_level - y_now
    else:
        y = y_level

    y_pred = model.predict(X)

    mae = float(mean_absolute_error(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    rmse = float(np.sqrt(mse))
    mbe = float((y_pred - y).mean())

    # Avoid copying the complete evaluation DataFrame.
    station_errors = pd.DataFrame(
        {
            "station_name": df["station_name"].astype(str).to_numpy(),
            "error": (y_pred - y).abs(),
        }
    )

    mae_by_station = (
        station_errors
        .groupby("station_name")["error"]
        .mean()
        .reset_index()
        .rename(columns={"error": "mae"})
    )

    mae_by_station_list = [
        {
            "station_name": row["station_name"],
            "mae": float(row["mae"]),
        }
        for _, row in mae_by_station.iterrows()
    ]

    return {
        "mae": mae,
        "rmse": rmse,
        "mbe": mbe,
        "mae_by_station": mae_by_station_list,
    }


def write_registry_record(summary: dict, gcs_model_path: str, metrics: dict, status: str = "staging"):
    row = {
        "model_version": summary["model_version"],
        "model_type": "gauge24h",
        "gcs_path": gcs_model_path,
        "training_table": summary["table_name"],
        "train_start_utc": summary["train_start_utc"],
        "train_end_utc": summary["train_end_utc"],
        "rows_trained": summary["rows_trained"],
        "stations_trained": summary["stations_trained"],
        "target_column": summary["target_column"],
        "target_mode": summary["target_mode"],
        "horizon_hours": summary["horizon_hours"],
        "trained_at_utc": summary["trained_at_utc"],
        "status": status,
        "promoted_at_utc": None,
        "evaluation_metrics_json": json.dumps(metrics),
        "notes": "",
    }
    job = bq_client.load_table_from_json([row], REGISTRY_TABLE)
    job.result()


def write_evaluation_record(model_version: str, metrics: dict):
    row = {
        "model_version": model_version,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": EVAL_SPLIT_NAME,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mbe": metrics["mbe"],
        "mae_by_station": metrics["mae_by_station"],
        "metrics_json": json.dumps(metrics),
    }
    job = bq_client.load_table_from_json([row], EVAL_TABLE)
    job.result()


def main():
    # --- Training ---
    df, target_column, feature_columns = load_training_frame()
    validate_training_frame(df, target_column, feature_columns)

    X = df[feature_columns].copy()
    y = build_training_target(df, target_column)

    model = build_pipeline(feature_columns)
    model.fit(X, y)

    train_start = df["timestamp_utc"].min()
    train_end = df["timestamp_utc"].max()
    model_version = stable_model_version(train_start, train_end, len(df), target_column)

    joblib.dump(model, MODEL_PATH)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "table_name": TRAIN_TABLE_NAME,
        "train_split_used": TRAIN_SPLIT_NAME,
        "target_column": target_column,
        "target_mode": TARGET_MODE,
        "horizon_hours": HORIZON_HOURS,
        "feature_columns_requested": PRODUCTION_FEATURE_COLUMNS,
        "feature_columns_used": feature_columns,
        "rows_trained": int(len(df)),
        "stations_trained": int(df["station_name"].nunique()),
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "model_path": str(MODEL_PATH),
        "max_feature_null_fraction": MAX_FEATURE_NULL_FRACTION,
        "null_profile": training_null_profile(df, feature_columns, target_column),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Upload to GCS ---
    gcs_model_path = upload_blob(
        MODEL_PATH,
        f"models/staging/{model_version}/model.joblib"
    )
    gcs_summary_path = upload_json(
        summary,
        f"artifacts/{model_version}/training_summary.json"
    )

    # --- Evaluation ---
    # Release training data before loading validation data.
    del X
    del y
    del df

    import gc
    gc.collect()

    # --- Evaluation ---
    metrics = evaluate_model(
        model,
        feature_columns,
        target_column,
    )
    gcs_metrics_path = upload_json(
        metrics,
        f"artifacts/{model_version}/evaluation_metrics.json"
    )

    # --- Write registry + eval records ---
    write_registry_record(summary, gcs_model_path, metrics, status="staging")
    write_evaluation_record(model_version, metrics)

    print("=== Training & registration complete ===")
    print(f"Model version: {model_version}")
    print(f"GCS model: {gcs_model_path}")
    print(f"GCS summary: {gcs_summary_path}")
    print(f"GCS metrics: {gcs_metrics_path}")
    print(f"MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, MBE: {metrics['mbe']:.3f}")


if __name__ == "__main__":
    main()