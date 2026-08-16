from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
from modeling.data_loader import load_bigquery_table, write_bigquery_table

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CURATED_DATASET = os.getenv("CURATED_DATASET", "rhein_curated").strip()
EVAL_SPLIT_NAME = os.getenv("GAUGE24H_EVAL_SPLIT_NAME", "test").strip().lower()
if EVAL_SPLIT_NAME not in {"production", "validation", "test"}:
    raise ValueError("GAUGE24H_EVAL_SPLIT_NAME must be production, validation, or test")

PREDICTIONS_HISTORY_TABLE = os.getenv("PREDICTIONS_HISTORY_TABLE", "gauge_24h_prediction_history").strip()
PREDICTIONS_TEST_TABLE = os.getenv("PREDICTIONS_TEST_TABLE", "gauge_24h_production_predictions_test").strip()
PREDICTIONS_VALIDATION_TABLE = os.getenv("PREDICTIONS_VALIDATION_TABLE", "gauge_24h_production_predictions_validation").strip()
ACTUALS_TABLE = os.getenv("GAUGE24H_ACTUALS_TABLE", "pegelonline_measurements_curated").strip()
DATASET_SPLITS_TABLE = os.getenv("GAUGE24H_DATASET_SPLITS_TABLE", "dataset_splits_gauge_24h").strip()
EVALUATION_TABLE = os.getenv("EVALUATIONS_TABLE", "gauge_24h_prediction_evaluations" if EVAL_SPLIT_NAME == "production" else f"gauge_24h_{EVAL_SPLIT_NAME}_prediction_evaluations").strip()
THRESHOLD_CONFIG_PATH = Path(os.getenv("GAUGE24H_THRESHOLD_CONFIG_PATH", "config/thresholds.yaml").strip())
ACTUAL_MATCH_TOLERANCE_MINUTES = int(os.getenv("GAUGE24H_ACTUAL_MATCH_TOLERANCE_MINUTES", "30"))
PREDICTIONS_TABLE = {"production": PREDICTIONS_HISTORY_TABLE, "test": PREDICTIONS_TEST_TABLE, "validation": PREDICTIONS_VALIDATION_TABLE}[EVAL_SPLIT_NAME]

EVALUATION_KEY_COLUMNS = ["split_name", "run_id", "station_name", "forecast_timestamp_utc", "model_version"]
FINAL_EVALUATION_COLUMNS = [
    "run_id", "model_version", "split_name", "station_name", "timeseries_name", "unit", "prediction_unit", "source",
    "timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc", "prediction_horizon_hours", "target_mode", "target_column",
    "prediction", "actual_value", "actual_available", "actual_timestamp_utc", "actual_match_tolerance_minutes",
    "error", "absolute_error", "squared_error", "ape", "threshold", "actual_event_low_water", "pred_event_low_water", "evaluated_at_utc",
]
EVALUATION_CSV = OUTPUT_DIR / f"gauge_24h_prediction_evaluation_{EVAL_SPLIT_NAME}.csv"
EVALUATION_SUMMARY_JSON = OUTPUT_DIR / f"gauge_24h_prediction_evaluation_summary_{EVAL_SPLIT_NAME}.json"
STATION_METRICS_CSV = OUTPUT_DIR / f"gauge_24h_prediction_evaluation_station_metrics_{EVAL_SPLIT_NAME}.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / f"gauge_24h_prediction_evaluation_model_metrics_{EVAL_SPLIT_NAME}.csv"


def normalize_station_name(value: object) -> str:
    return str(value).strip().upper()


def normalize_merge_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def load_thresholds() -> dict[str, float]:
    if not THRESHOLD_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Threshold config not found: {THRESHOLD_CONFIG_PATH}")
    import yaml
    with THRESHOLD_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    values = data.get("low_water_thresholds_cm", data)
    if not isinstance(values, dict):
        raise ValueError("Threshold config must contain low_water_thresholds_cm")
    result = {normalize_station_name(k): float(v) for k, v in values.items() if v is not None}
    if not result:
        raise ValueError("No thresholds found in threshold config")
    return result


THRESHOLD_BY_STATION = load_thresholds()


def load_predictions() -> pd.DataFrame:
    df = load_bigquery_table(PREDICTIONS_TABLE, dataset=CURATED_DATASET, order_by="split_name, forecast_timestamp_utc, station_name")
    if df.empty:
        raise ValueError(f"No rows found in {CURATED_DATASET}.{PREDICTIONS_TABLE}")
    required = {"station_name", "prediction", "split_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Predictions table missing required columns: {missing}")
    for column in ["timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc"]:
        if column in df.columns:
            df[column] = normalize_merge_timestamp(df[column])
    if "prediction_horizon_hours" in df.columns:
        df["prediction_horizon_hours"] = pd.to_numeric(df["prediction_horizon_hours"], errors="coerce")
    if "forecast_timestamp_utc" not in df.columns:
        if "timestamp_utc" not in df.columns:
            raise ValueError("Predictions table must include timestamp_utc or forecast_timestamp_utc")
        horizon = 24
        values = df.get("prediction_horizon_hours", pd.Series(dtype=float)).dropna().unique()
        if len(values):
            horizon = int(values[0])
        df["forecast_timestamp_utc"] = normalize_merge_timestamp(df["timestamp_utc"] + pd.to_timedelta(horizon, unit="h"))
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["split_name"] = df["split_name"].astype("string").str.lower()
    df["station_name"] = df["station_name"].astype("string")
    return df.dropna(subset=["station_name", "prediction", "forecast_timestamp_utc"]).copy()


def load_split_actuals() -> pd.DataFrame:
    df = load_bigquery_table(DATASET_SPLITS_TABLE, dataset=CURATED_DATASET, order_by="split_name, timestamp_utc, station_name")
    if df.empty:
        raise ValueError(f"No rows found in {CURATED_DATASET}.{DATASET_SPLITS_TABLE}")
    required = {"split_name", "station_name", "timestamp_utc"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset splits table missing required columns: {missing}")
    df["timestamp_utc"] = normalize_merge_timestamp(df["timestamp_utc"])
    df["station_name"] = df["station_name"].astype("string")
    df["split_name"] = df["split_name"].astype("string").str.lower()
    for column in [c for c in df.columns if c.startswith("target_value_t_plus_")] + ["target_value"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_production_actuals() -> pd.DataFrame:
    df = load_bigquery_table(ACTUALS_TABLE, dataset=CURATED_DATASET, columns=["station_name", "timestamp_utc", "value", "unit", "source"], order_by="timestamp_utc, station_name", allow_missing_columns=True)
    if df.empty:
        raise ValueError(f"No rows found in {CURATED_DATASET}.{ACTUALS_TABLE}")
    required = {"station_name", "timestamp_utc", "value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Production actuals table missing required columns: {missing}")
    df["station_name"] = df["station_name"].astype("string")
    df["timestamp_utc"] = normalize_merge_timestamp(df["timestamp_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["station_name", "timestamp_utc", "value"]).sort_values(["station_name", "timestamp_utc"])
    return df.drop_duplicates(["station_name", "timestamp_utc"], keep="last").reset_index(drop=True)


def resolve_actual_values(predictions: pd.DataFrame, actuals: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in actuals.columns:
        raise ValueError(f"Target column {target_column!r} not found in actuals table")
    predictions = predictions.copy()
    actuals = actuals.copy()
    predictions["forecast_timestamp_utc"] = normalize_merge_timestamp(predictions["forecast_timestamp_utc"])
    actuals["timestamp_utc"] = normalize_merge_timestamp(actuals["timestamp_utc"])
    actuals = actuals[["split_name", "station_name", "timestamp_utc", target_column]].rename(columns={"timestamp_utc": "actual_timestamp_utc", target_column: "actual_value"})
    actuals["target_column"] = target_column
    return predictions.merge(actuals, left_on=["split_name", "station_name", "forecast_timestamp_utc", "target_column"], right_on=["split_name", "station_name", "actual_timestamp_utc", "target_column"], how="left", validate="many_to_one")


def resolve_production_actual_values(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    left = predictions.copy()
    right = actuals.copy()
    left["forecast_timestamp_utc"] = normalize_merge_timestamp(left["forecast_timestamp_utc"])
    right["timestamp_utc"] = normalize_merge_timestamp(right["timestamp_utc"])
    right = right.rename(columns={"value": "actual_value", "timestamp_utc": "actual_timestamp_utc"})
    left = left.dropna(subset=["station_name", "forecast_timestamp_utc"])
    right = right.dropna(subset=["station_name", "actual_timestamp_utc", "actual_value"])
    groups = []
    for station, station_predictions in left.groupby("station_name", dropna=False):
        station_actuals = right[right["station_name"] == station].copy()
        if station_actuals.empty:
            group = station_predictions.copy()
            group["actual_timestamp_utc"] = pd.NaT
            group["actual_value"] = np.nan
        else:
            station_predictions = station_predictions.sort_values("forecast_timestamp_utc").reset_index(drop=True)
            station_actuals = station_actuals.sort_values("actual_timestamp_utc").reset_index(drop=True)
            station_predictions["forecast_timestamp_utc"] = normalize_merge_timestamp(station_predictions["forecast_timestamp_utc"])
            station_actuals["actual_timestamp_utc"] = normalize_merge_timestamp(station_actuals["actual_timestamp_utc"])
            group = pd.merge_asof(station_predictions, station_actuals, left_on="forecast_timestamp_utc", right_on="actual_timestamp_utc", direction="nearest", tolerance=pd.Timedelta(minutes=ACTUAL_MATCH_TOLERANCE_MINUTES), suffixes=("", "_actual_source"))
        groups.append(group)
    result = pd.concat(groups, ignore_index=True) if groups else left.copy()
    result["actual_value"] = result.get("actual_value", np.nan)
    result["actual_timestamp_utc"] = result.get("actual_timestamp_utc", pd.NaT)
    result["actual_available"] = result["actual_value"].notna()
    result["actual_match_tolerance_minutes"] = ACTUAL_MATCH_TOLERANCE_MINUTES
    return result


def prepare_evaluation_frame(predictions: pd.DataFrame, actuals: pd.DataFrame, split_name: str) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    predictions = predictions[predictions["split_name"] == split_name].copy()
    if predictions.empty:
        raise RuntimeError(f"No predictions found for split={split_name!r}")
    if split_name == "production":
        mature = predictions[predictions["forecast_timestamp_utc"] <= now].copy()
        if mature.empty:
            raise RuntimeError("No matured production predictions yet; the forecast horizon has not elapsed")
        merged = resolve_production_actual_values(mature, actuals)
    else:
        if "prediction_ready_utc" in predictions.columns:
            predictions = predictions[predictions["prediction_ready_utc"] == predictions["prediction_ready_utc"].max()].copy()
        mature = predictions[predictions["forecast_timestamp_utc"] <= now].copy()
        if mature.empty:
            raise RuntimeError("No matured predictions yet; the forecast horizon has not elapsed")
        if "target_column" not in mature.columns:
            raise RuntimeError("Predictions do not have a target_column field")
        target_column = str(mature["target_column"].iloc[0])
        merged = resolve_actual_values(mature, actuals[actuals["split_name"] == split_name], target_column)
    if not merged["actual_value"].notna().any():
        raise RuntimeError(f"No actuals matched within {ACTUAL_MATCH_TOLERANCE_MINUTES} minutes")
    merged["actual_available"] = merged["actual_value"].notna()
    merged = merged[merged["actual_available"]].copy()
    merged["error"] = merged["actual_value"] - merged["prediction"]
    merged["abs_error"] = merged["error"].abs()
    merged["absolute_error"] = merged["abs_error"]
    merged["squared_error"] = merged["error"] ** 2
    merged["ape"] = np.where(merged["actual_value"].abs() > 1e-9, merged["abs_error"] / merged["actual_value"].abs(), np.nan)
    merged["station_name"] = merged["station_name"].map(normalize_station_name)
    merged["threshold"] = merged["station_name"].map(THRESHOLD_BY_STATION)
    merged["actual_event_low_water"] = np.where(merged["threshold"].notna(), merged["actual_value"] <= merged["threshold"], pd.NA)
    merged["pred_event_low_water"] = np.where(merged["threshold"].notna(), merged["prediction"] <= merged["threshold"], pd.NA)
    merged["evaluated_at_utc"] = pd.Timestamp.now(tz="UTC")
    return merged.sort_values(["forecast_timestamp_utc", "station_name"]).reset_index(drop=True)


def select_final_evaluation_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["run_id", "model_version", "split_name", "station_name", "forecast_timestamp_utc", "prediction", "actual_value", "actual_available", "error", "evaluated_at_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Evaluation output missing required columns: {missing}")
    result = df[[c for c in FINAL_EVALUATION_COLUMNS if c in df.columns]].copy()
    for column in ["timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc", "actual_timestamp_utc", "evaluated_at_utc"]:
        if column in result.columns:
            result[column] = normalize_merge_timestamp(result[column])
    return result


def regression_metrics(df: pd.DataFrame) -> dict:
    # abs_error is an internal calculation column and is deliberately removed
    # from the final table. Use absolute_error as the persisted canonical field.
    absolute = pd.to_numeric(df["absolute_error"], errors="coerce") if "absolute_error" in df.columns else pd.to_numeric(df["abs_error"], errors="coerce")
    squared = pd.to_numeric(df["squared_error"], errors="coerce")
    errors = pd.to_numeric(df["error"], errors="coerce")
    ape = pd.to_numeric(df["ape"], errors="coerce")
    if df.empty:
        return {"rows_evaluated": 0, "mae": None, "rmse": None, "bias": None, "mape": None, "p90_abs_error": None}
    return {"rows_evaluated": int(len(df)), "mae": float(absolute.mean()), "rmse": float(np.sqrt(squared.mean())), "bias": float(errors.mean()), "mape": float(ape.dropna().mean()) if ape.notna().any() else None, "p90_abs_error": float(absolute.quantile(.9))}


def classification_metrics(df: pd.DataFrame) -> dict:
    data = df.dropna(subset=["actual_event_low_water", "pred_event_low_water"])
    if data.empty:
        return {"event_rows": 0, "event_precision": None, "event_recall": None, "event_f1": None}
    actual = data["actual_event_low_water"].astype(bool)
    predicted = data["pred_event_low_water"].astype(bool)
    tp, fp, fn = int((actual & predicted).sum()), int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"event_rows": int(len(data)), "event_precision": precision, "event_recall": recall, "event_f1": f1}


def build_station_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in df.groupby("station_name", dropna=False):
        row = {"station_name": station}
        row.update(regression_metrics(group))
        row.update(classification_metrics(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("rmse", ascending=False).reset_index(drop=True) if not result.empty else result


def build_model_metrics(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    columns = [c for c in ["model_version", "run_id", "target_mode", "prediction_horizon_hours", "target_column"] if c in df.columns]
    rows = []
    if not columns:
        return pd.DataFrame()
    for keys, group in df.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {"split_name": split_name, **dict(zip(columns, keys))}
        row.update(regression_metrics(group))
        row.update(classification_metrics(group))
        row["forecast_min_utc"] = group["forecast_timestamp_utc"].min().isoformat()
        row["forecast_max_utc"] = group["forecast_timestamp_utc"].max().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def merge_evaluation_results(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    combined = new.copy() if existing.empty else existing.copy() if new.empty else pd.concat([existing, new], ignore_index=True, sort=False)
    combined = select_final_evaluation_columns(combined)
    keys = [c for c in EVALUATION_KEY_COLUMNS if c in combined.columns]
    if not keys:
        raise ValueError("Cannot deduplicate evaluation results: no key columns are present")
    before = len(combined)
    combined = combined.sort_values([c for c in ["evaluated_at_utc", "forecast_timestamp_utc"] if c in combined.columns]).drop_duplicates(keys, keep="last").reset_index(drop=True)
    return combined, before - len(combined)


def main() -> None:
    predictions = load_predictions()
    actuals = load_production_actuals() if EVAL_SPLIT_NAME == "production" else load_split_actuals()
    try:
        eval_df = prepare_evaluation_frame(predictions, actuals, EVAL_SPLIT_NAME)
    except RuntimeError as exc:
        print(f"Evaluation not ready: {exc}")
        sys.exit(0)
    eval_df = select_final_evaluation_columns(eval_df)
    try:
        existing = load_bigquery_table(EVALUATION_TABLE, dataset=CURATED_DATASET, order_by="split_name, forecast_timestamp_utc, station_name")
        existing = select_final_evaluation_columns(existing) if not existing.empty else existing
    except Exception:
        existing = pd.DataFrame()
    combined, duplicates_removed = merge_evaluation_results(existing, eval_df)
    write_bigquery_table(combined, EVALUATION_TABLE, dataset=CURATED_DATASET, if_exists="replace")
    station_metrics = build_station_metrics(eval_df)
    model_metrics = build_model_metrics(eval_df, EVAL_SPLIT_NAME)
    summary = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "ok", "split_name": EVAL_SPLIT_NAME, "source_predictions_table": PREDICTIONS_TABLE, "source_actuals_table": ACTUALS_TABLE if EVAL_SPLIT_NAME == "production" else DATASET_SPLITS_TABLE, "actual_match_tolerance_minutes": ACTUAL_MATCH_TOLERANCE_MINUTES, **regression_metrics(eval_df), **classification_metrics(eval_df), "stations_evaluated": int(eval_df["station_name"].nunique()), "new_evaluation_rows": int(len(eval_df)), "total_evaluation_rows_after_merge": int(len(combined)), "duplicate_rows_removed": int(duplicates_removed), "final_columns": FINAL_EVALUATION_COLUMNS}
    eval_df.to_csv(EVALUATION_CSV, index=False)
    station_metrics.to_csv(STATION_METRICS_CSV, index=False)
    model_metrics.to_csv(MODEL_METRICS_CSV, index=False)
    with EVALUATION_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("=== Gauge 24h evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote BigQuery table: {CURATED_DATASET}.{EVALUATION_TABLE}")


if __name__ == "__main__":
    main()