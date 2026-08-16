from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

import joblib
import numpy as np
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

PREDICTIONS_CSV = OUTPUT_DIR / "gauge_24h_production_predictions.csv"
PREDICTIONS_SUMMARY_JSON = OUTPUT_DIR / "gauge_24h_production_predictions_summary.json"
PREDICTIONS_TABLE_BASE = "gauge_24h_production_predictions"

# BigQuery Environment Variables
CURATED_DATASET = os.getenv("CURATED_DATASET", "rhein_curated").strip()
PREDICTIONS_HISTORY_TABLE = os.getenv("PREDICTIONS_HISTORY_TABLE", "gauge_24h_prediction_history").strip()
ACTUALS_TABLE = os.getenv("GAUGE24H_ACTUALS_TABLE", "pegelonline_measurements_curated").strip()

WEATHER_REQUIRED_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
]

PRED_SPLIT_ENV = os.getenv("GAUGE24H_PRED_SPLIT", "test").strip().lower()
if PRED_SPLIT_ENV not in {"validation", "test", "production"}:
    raise ValueError("GAUGE24H_PRED_SPLIT must be 'validation', 'test', or 'production'")

HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
ACTUAL_MATCH_TOLERANCE_MINUTES = int(os.getenv("GAUGE24H_ACTUAL_MATCH_TOLERANCE_MINUTES", "30"))


def normalize_timestamp(series_or_scalar) -> pd.Series | pd.Timestamp:
    """Ensure standard BigQuery microsecond UTC timestamp format for series or scalars."""
    if isinstance(series_or_scalar, pd.Series):
        parsed = pd.to_datetime(series_or_scalar, utc=True, errors="coerce")
        return parsed.dt.floor("min").astype("datetime64[us, UTC]")
    else:
        parsed = pd.to_datetime(series_or_scalar, utc=True, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        return parsed.floor("min").tz_convert("UTC")


def make_run_id(model_version: str, split_name: str) -> str:
    seed = f"{model_version}|{split_name}|{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def load_production_artifacts() -> tuple[dict, object, str]:
    registry_df = load_bigquery_table(
        "model_registry",
        dataset="mlops",
        columns=["model_version", "status", "gcs_path", "promoted_at_utc"],
        where_sql="status = 'prod'",
        order_by="promoted_at_utc DESC",
        allow_missing_columns=False,
    )
    if registry_df.empty:
        raise RuntimeError("No production model found in mlops.model_registry.")

    row = registry_df.iloc[0]
    model_version = str(row["model_version"])
    model_gcs_path = str(row["gcs_path"])
    if not model_gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid production model GCS path: {model_gcs_path}")

    bucket_name = model_gcs_path[5:].split("/", 1)[0]
    summary_gcs_path = f"gs://{bucket_name}/artifacts/{model_version}/training_summary.json"
    model_path = OUTPUT_DIR / f"model_{model_version}.joblib"
    download_blob(model_gcs_path, model_path)
    training_summary = download_json(summary_gcs_path)

    if training_summary.get("model_version") != model_version:
        raise RuntimeError("Model/summary version mismatch")

    return training_summary, joblib.load(model_path), model_version


def resolve_feature_columns(training_summary: dict, df: pd.DataFrame) -> list[str]:
    trained = training_summary.get("feature_columns_used") or training_summary.get("feature_columns") or []
    if trained:
        return [column for column in trained if column in df.columns]
    return [column for column in PRODUCTION_FEATURE_COLUMNS if column in df.columns]


def prepare_inference_frame(
    horizon_hours: int,
    training_summary: dict,
) -> tuple[pd.DataFrame, str, list[str]]:
    target_column = f"target_value_t_plus_{horizon_hours}h"
    requested = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS
            + [
                DEFAULT_TARGET_COLUMN,
                target_column,
                "target_value",
                "timestamp_utc",
                "split_name",
                "station_name",
            ]
        )
    )

    df = load_bigquery_table(
        TRAIN_TABLE_NAME,
        columns=requested,
        order_by="timestamp_utc, station_name",
        allow_missing_columns=True,
    )

    missing_required = [
        column
        for column in ROBUSTNESS_REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_required:
        raise ValueError(f"Missing required inference base columns: {missing_required}")
    if "split_name" not in df.columns or "station_name" not in df.columns:
        raise ValueError("Inference table must include split_name and station_name")

    df["timestamp_utc"] = normalize_timestamp(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "station_name"]).copy()
    df["split_name"] = df["split_name"].astype("string").str.lower()
    df = df[df["split_name"] == PRED_SPLIT_ENV].copy()

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string")

    numeric_columns = list(
        dict.fromkeys(NUMERIC_COLUMNS_LEAN + [DEFAULT_TARGET_COLUMN, target_column, "target_value"])
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    feature_columns = resolve_feature_columns(training_summary, df)
    trained_features = training_summary.get("feature_columns_used") or training_summary.get("feature_columns") or []
    missing_trained = [column for column in trained_features if column not in df.columns]
    if missing_trained:
        raise ValueError(f"Inference data is missing trained feature columns: {missing_trained}")
    if not feature_columns:
        raise ValueError("No usable feature columns available for inference")

    required_complete = list(
        dict.fromkeys(
            ["station_name", "timestamp_utc", "target_value"]
            + WEATHER_REQUIRED_COLUMNS
            + feature_columns
        )
    )
    missing_complete_columns = [column for column in required_complete if column not in df.columns]
    if missing_complete_columns:
        raise ValueError(f"Required complete-data columns are missing: {missing_complete_columns}")

    df = df.dropna(subset=required_complete).copy()
    if df.empty:
        raise ValueError(f"No complete rows available for split '{PRED_SPLIT_ENV}'")

    if PRED_SPLIT_ENV == "production":
        latest_rows = (
            df.groupby(["split_name", "station_name"], dropna=False)["timestamp_utc"]
            .max()
            .rename("latest_timestamp_utc")
            .reset_index()
        )
        df = df.merge(
            latest_rows,
            on=["split_name", "station_name"],
            how="inner",
            validate="many_to_one",
        )
        df = df[df["timestamp_utc"] == df["latest_timestamp_utc"]].copy()
        df = df.drop(columns=["latest_timestamp_utc"])

    df = df.sort_values(["split_name", "station_name", "timestamp_utc"]).reset_index(drop=True)
    df["forecast_timestamp_utc"] = normalize_timestamp(df["timestamp_utc"] + pd.to_timedelta(horizon_hours, unit="h"))
    df["prediction_ready_utc"] = normalize_timestamp(pd.Timestamp.now(tz="UTC"))
    return df, target_column, feature_columns


def validate_inference_frame(df: pd.DataFrame, feature_columns: list[str]) -> None:
    required = ["timestamp_utc", "target_value", "split_name", "station_name"] + feature_columns
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")
    if df.empty:
        raise ValueError("Inference frame is empty")
    if df[required].isna().any().any():
        raise ValueError("Inference frame contains null required values")
    if PRED_SPLIT_ENV == "production":
        duplicates = df.duplicated(subset=["split_name", "station_name"], keep=False)
        if duplicates.any():
            raise ValueError("Production inference must contain one row per station")


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
    pred_raw = pd.to_numeric(
        model.predict(result[feature_columns].copy()),
        errors="coerce",
    )

    result["prediction_delta"] = pd.NA
    if target_mode == "delta":
        result["prediction_delta"] = pred_raw
        result["prediction_level"] = pred_raw + pd.to_numeric(
            result["target_value"], errors="coerce"
        ).to_numpy()
    elif target_mode == "level":
        result["prediction_level"] = pred_raw
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")

    result["prediction_level"] = pd.to_numeric(
        result["prediction_level"], errors="coerce"
    ).round(1)
    result["prediction"] = result["prediction_level"]

    if target_mode == "delta":
        result["prediction_delta"] = pd.to_numeric(
            result["prediction_delta"], errors="coerce"
        ).round(1)

    run_id = os.getenv("MLOPS_RUN_ID") or make_run_id(model_version, PRED_SPLIT_ENV)
    result["model_version"] = model_version
    result["run_id"] = run_id
    result["target_column"] = target_column
    result["target_mode"] = target_mode
    result["prediction_horizon_hours"] = horizon_hours
    result["prediction_unit"] = result.get("unit", "cm")

    if PRED_SPLIT_ENV == "production":
        result["actual_if_available"] = pd.NA
        result["actual_available_now"] = False
    else:
        result["actual_if_available"] = result[target_column]
        result["actual_available_now"] = result[target_column].notna()

    output_cols = list(
        dict.fromkeys(
            [
                "run_id",
                "model_version",
                "split_name",
                "station_name",
                "timeseries_name",
                "unit",
                "prediction_unit",
                "source",
                "timestamp_utc",
                "forecast_timestamp_utc",
                "prediction_ready_utc",
                "prediction_horizon_hours",
                "target_mode",
                "target_column",
                "prediction_delta",
                "prediction_level",
                "prediction",
                "actual_if_available",
                "actual_available_now",
            ]
            + feature_columns
        )
    )
    output_cols = [column for column in output_cols if column in result.columns]

    for column in ["timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc"]:
        if column in result.columns:
            result[column] = normalize_timestamp(result[column])

    return result[output_cols].sort_values(
        ["split_name", "station_name", "timestamp_utc"]
    ).reset_index(drop=True)


def load_actuals_for_history() -> pd.DataFrame:
    df = load_bigquery_table(
        ACTUALS_TABLE,
        dataset=CURATED_DATASET,
        columns=["station_name", "timestamp_utc", "value"],
        order_by="station_name, timestamp_utc",
        allow_missing_columns=True,
    )

    if df.empty or not {"station_name", "timestamp_utc", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["station_name", "actual_timestamp_utc", "actual_value"])

    df["station_name"] = df["station_name"].astype("string").str.strip()
    df["timestamp_utc"] = normalize_timestamp(df["timestamp_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["station_name", "timestamp_utc", "value"])
    df = df.sort_values(["station_name", "timestamp_utc"]).drop_duplicates(
        ["station_name", "timestamp_utc"], keep="last"
    )

    return df.rename(
        columns={"timestamp_utc": "actual_timestamp_utc", "value": "actual_value"}
    )[["station_name", "actual_timestamp_utc", "actual_value"]]


def load_existing_history() -> pd.DataFrame:
    try:
        history = load_bigquery_table(
            PREDICTIONS_HISTORY_TABLE,
            dataset=CURATED_DATASET,
            order_by=None,
            allow_missing_columns=True,
        )
    except Exception as exc:
        if any(m in str(exc).lower() for m in ["not found", "does not exist", "404", "tableid"]):
            return pd.DataFrame()
        raise

    return history


def refresh_history_actuals(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    result = history.copy()
    if "station_name" not in result.columns:
        result["station_name"] = pd.Series(pd.NA, index=result.index, dtype="string")

    result["station_name"] = result["station_name"].astype("string").str.strip()
    result["forecast_timestamp_utc"] = normalize_timestamp(result["forecast_timestamp_utc"])

    actuals = load_actuals_for_history()
    if actuals.empty:
        return result

    has_station_name = result["station_name"].notna() & result["station_name"].ne("")
    legacy_rows = result.loc[~has_station_name].copy()
    station_rows = result.loc[has_station_name].copy()

    if station_rows.empty:
        return result

    groups = []
    for station_name, station_history in station_rows.groupby("station_name", dropna=False):
        station_history = station_history.sort_values("forecast_timestamp_utc").reset_index(drop=True)
        station_actuals = actuals[actuals["station_name"] == station_name].sort_values("actual_timestamp_utc").reset_index(drop=True)

        if station_actuals.empty:
            groups.append(station_history)
            continue

        station_history["forecast_timestamp_utc"] = normalize_timestamp(station_history["forecast_timestamp_utc"])
        station_actuals["actual_timestamp_utc"] = normalize_timestamp(station_actuals["actual_timestamp_utc"])

        matched = pd.merge_asof(
            station_history,
            station_actuals.drop(columns=["station_name"]),
            left_on="forecast_timestamp_utc",
            right_on="actual_timestamp_utc",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=ACTUAL_MATCH_TOLERANCE_MINUTES),
        )
        groups.append(matched)

    refreshed_station_rows = pd.concat(groups, ignore_index=True)

    if "actual_value" in refreshed_station_rows.columns:
        existing_actual = pd.to_numeric(refreshed_station_rows.get("actual_if_available", pd.NA), errors="coerce")
        existing_flag = refreshed_station_rows.get("actual_available_now", pd.Series(False, index=refreshed_station_rows.index)).fillna(False).astype(bool)
        
        matched_actual = refreshed_station_rows["actual_value"].notna()
        refreshed_station_rows["actual_if_available"] = refreshed_station_rows["actual_value"].where(matched_actual, existing_actual)
        refreshed_station_rows["actual_available_now"] = matched_actual | existing_flag
        
        refreshed_station_rows = refreshed_station_rows.drop(columns=["actual_value", "actual_timestamp_utc"], errors="ignore")

    return pd.concat([legacy_rows, refreshed_station_rows], ignore_index=True, sort=False).reset_index(drop=True)


def deduplicate_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
        
    result = history.copy()
    has_station_name = result["station_name"].notna() & result["station_name"].ne("")
    legacy_rows = result.loc[~has_station_name].copy()
    station_rows = result.loc[has_station_name].copy()

    if station_rows.empty:
        return legacy_rows.reset_index(drop=True)

    deduplication_keys = ["station_name", "forecast_timestamp_utc", "model_version"]
    sort_columns = ["station_name", "forecast_timestamp_utc"]
    
    if "prediction_ready_utc" in station_rows.columns:
        sort_columns.append("prediction_ready_utc")

    station_rows = station_rows.sort_values(sort_columns).drop_duplicates(subset=deduplication_keys, keep="last")

    return pd.concat([legacy_rows, station_rows], ignore_index=True, sort=False).reset_index(drop=True)


def build_summary(
    pred_df: pd.DataFrame,
    training_summary: dict,
    feature_columns: list[str],
    current_table: str,
) -> dict:
    forecasts = pd.to_datetime(pred_df["forecast_timestamp_utc"], utc=True, errors="coerce")
    return {
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": pred_df["run_id"].iloc[0] if len(pred_df) else None,
        "model_version": pred_df["model_version"].iloc[0] if len(pred_df) else None,
        "target_column": training_summary.get("target_column"),
        "target_mode": training_summary.get("target_mode"),
        "horizon_hours": training_summary.get("horizon_hours"),
        "prediction_semantics": "absolute_level_in_prediction; raw_delta_in_prediction_delta" if training_summary.get("target_mode") == "delta" else "absolute_level",
        "training_rows": training_summary.get("rows_trained"),
        "feature_columns_used_for_inference": feature_columns,
        "rows_predicted": int(len(pred_df)),
        "stations_predicted": int(pred_df["station_name"].nunique()) if len(pred_df) else 0,
        "split_present": PRED_SPLIT_ENV,
        "source_table": TRAIN_TABLE_NAME,
        "prediction_table": current_table,
        "history_table": PREDICTIONS_HISTORY_TABLE if PRED_SPLIT_ENV == "production" else None,
        "min_forecast_timestamp_utc": forecasts.min().isoformat() if forecasts.notna().any() else None,
        "max_forecast_timestamp_utc": forecasts.max().isoformat() if forecasts.notna().any() else None,
    }


def main() -> dict:
    training_summary, model, model_version = load_production_artifacts()
    target_mode = training_summary.get("target_mode", "level")
    inference_df, target_column, feature_columns = prepare_inference_frame(
        HORIZON_HOURS,
        training_summary,
    )
    validate_inference_frame(inference_df, feature_columns)

    pred_df = score_frame(
        inference_df,
        model,
        model_version,
        target_mode,
        HORIZON_HOURS,
        target_column,
        feature_columns,
    )

    current_table = (
        PREDICTIONS_TABLE_BASE
        if PRED_SPLIT_ENV == "production"
        else f"{PREDICTIONS_TABLE_BASE}_{PRED_SPLIT_ENV}"
    )
    
    if PRED_SPLIT_ENV == "production":
        existing_history = load_existing_history()
        
        # Pull 1-day old predictions before overwriting the production table
        try:
            previous_predictions = load_bigquery_table(
                current_table,
                dataset=CURATED_DATASET,
                order_by=None,
                allow_missing_columns=True,
            )
        except Exception as exc:
            if any(m in str(exc).lower() for m in ["not found", "does not exist", "404", "tableid"]):
                previous_predictions = pd.DataFrame()
            else:
                raise
        
        if existing_history.empty:
            history = previous_predictions.copy()
        elif not previous_predictions.empty:
            history = pd.concat([existing_history, previous_predictions], ignore_index=True, sort=False)
        else:
            history = existing_history.copy()
            
        history = deduplicate_history(history)
        history = refresh_history_actuals(history)
        
        # Explicitly enforce writing ONLY rows where the actual value has become available
        if not history.empty and "actual_available_now" in history.columns:
            history = history[history["actual_available_now"] == True].copy()
            
        history = deduplicate_history(history)

        if not history.empty:
            write_bigquery_table(
                history,
                PREDICTIONS_HISTORY_TABLE,
                dataset=CURATED_DATASET,
                if_exists="replace",
            )

    # Write current day predictions to production table
    write_bigquery_table(
        pred_df,
        current_table,
        dataset=CURATED_DATASET,
        if_exists="replace",
    )

    summary = build_summary(pred_df, training_summary, feature_columns, current_table)
    with PREDICTIONS_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    display_preds = pred_df.copy()
    for col in ["timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc", "actual_timestamp_utc"]:
        if col in display_preds.columns:
            display_preds[col] = pd.to_datetime(display_preds[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            
    display_preds.to_csv(PREDICTIONS_CSV, index=False)
    print(display_preds.head(30).to_string(index=False))
    print(json.dumps(summary, indent=2))
    
    return summary


if __name__ == "__main__":
    main()