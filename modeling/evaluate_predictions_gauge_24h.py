from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

import numpy as np
import pandas as pd

from modeling.data_loader import load_bigquery_table, write_bigquery_table

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_TABLE = "gauge_24h_production_prediction_evaluation"

EVALUATION_CSV = OUTPUT_DIR / "gauge_24h_prediction_evaluation.csv"
EVALUATION_SUMMARY_JSON = OUTPUT_DIR / "gauge_24h_prediction_evaluation_summary.json"
STATION_METRICS_CSV = OUTPUT_DIR / "gauge_24h_prediction_evaluation_station_metrics.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / "gauge_24h_prediction_evaluation_model_metrics.csv"

THRESHOLD_BY_STATION = {
    "KAUB": 120,
    "MAXAU": 380,
    "KOBLENZ": 150,
    "DUISBURG-RUHRORT": 260,
    "EMMERICH": 140,
    "KÖLN": 180,
    "MAINZ": 170,
    "WORMS": 120,
    "SPEYER": 200,
    "BONN": 170,
    "DÜSSELDORF": 190,
    "REES": 160,
}

EVAL_SPLIT_NAME = os.getenv("GAUGE24H_EVAL_SPLIT_NAME", "test").strip().lower()
if EVAL_SPLIT_NAME not in {"validation", "test"}:
    raise ValueError("GAUGE24H_EVAL_SPLIT_NAME must be 'validation' or 'test'")

PREDICTIONS_TABLE = f"gauge_24h_production_predictions_{EVAL_SPLIT_NAME}"


def load_predictions() -> pd.DataFrame:
    df = load_bigquery_table(
        PREDICTIONS_TABLE,
        dataset="rhein_curated",
        order_by="split_name, timestamp_utc, station_name",
    )
    if df.empty:
        raise ValueError(f"No rows found in rhein_curated.{PREDICTIONS_TABLE}")

    required = ["station_name", "timestamp_utc", "prediction", "split_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions table missing required columns: {missing}")

    for col in ["timestamp_utc", "prediction_ready_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    horizon_hours = int(df["prediction_horizon_hours"].iloc[0]) if "prediction_horizon_hours" in df.columns else 24
    df["forecast_timestamp_utc"] = df["timestamp_utc"] + pd.to_timedelta(horizon_hours, unit="h")

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    if "prediction_horizon_hours" in df.columns:
        df["prediction_horizon_hours"] = pd.to_numeric(df["prediction_horizon_hours"], errors="coerce")

    df["split_name"] = df["split_name"].astype("string").str.lower()
    return df


def load_actuals() -> pd.DataFrame:
    df = load_bigquery_table(
        "dataset_splits_gauge_24h",
        dataset="rhein_curated",
        order_by="split_name, timestamp_utc, station_name",
    )
    if df.empty:
        raise ValueError("No rows found in rhein_curated.dataset_splits_gauge_24h")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for col in [c for c in df.columns if c.startswith("target_value_t_plus_")] + ["target_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "split_name" in df.columns:
        df["split_name"] = df["split_name"].astype("string").str.lower()

    return df


def resolve_actual_values(pred_df: pd.DataFrame, actual_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in actual_df.columns:
        raise ValueError(f"Target column {target_column!r} not found in actuals table.")

    actual_long = actual_df[["split_name", "station_name", "timestamp_utc", target_column]].copy()
    actual_long = actual_long.rename(
        columns={"timestamp_utc": "actual_timestamp_utc", target_column: "actual_value"}
    )
    actual_long["target_column"] = target_column

    merged = pred_df.merge(
        actual_long,
        left_on=["split_name", "station_name", "forecast_timestamp_utc", "target_column"],
        right_on=["split_name", "station_name", "actual_timestamp_utc", "target_column"],
        how="left",
        validate="many_to_one",
    )
    return merged


def format_timedelta_human(delta: pd.Timedelta | None) -> str | None:
    if delta is None or pd.isna(delta):
        return None

    total_seconds = int(max(delta.total_seconds(), 0))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def build_not_ready_summary(pred_df: pd.DataFrame, reason: str, split_name: str, target_column: str | None = None) -> dict:
    now_utc = pd.Timestamp.now(tz="UTC")

    forecast_min = pred_df["forecast_timestamp_utc"].min() if len(pred_df) else None
    forecast_max = pred_df["forecast_timestamp_utc"].max() if len(pred_df) else None

    horizon_hours = None
    if "prediction_horizon_hours" in pred_df.columns:
        horizon_values = pd.to_numeric(pred_df["prediction_horizon_hours"], errors="coerce").dropna()
        if not horizon_values.empty:
            horizon_hours = int(horizon_values.min())

    evaluation_available_at = None
    time_until_ready = None

    if forecast_max is not None and horizon_hours is not None:
        evaluation_available_at = forecast_max + pd.Timedelta(hours=horizon_hours)
        time_until_ready = evaluation_available_at - now_utc
    elif forecast_max is not None:
        evaluation_available_at = forecast_max
        time_until_ready = evaluation_available_at - now_utc

    human_wait = format_timedelta_human(time_until_ready) if time_until_ready is not None else None
    wait_seconds = max(int(time_until_ready.total_seconds()), 0) if time_until_ready is not None else None

    message = reason
    if evaluation_available_at is not None:
        message = f"{reason} Evaluation can be performed at {evaluation_available_at.isoformat()}."
        if human_wait is not None and wait_seconds and wait_seconds > 0:
            message += f" Time remaining: {human_wait}."

    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_ready",
        "split_name": split_name,
        "target_column": target_column,
        "reason": reason,
        "message": message,
        "source_predictions_table": PREDICTIONS_TABLE,
        "source_actuals_table": "dataset_splits_gauge_24h",
        "rows_in_predictions_table": int(len(pred_df)),
        "model_versions_present": int(pred_df["model_version"].nunique()) if "model_version" in pred_df.columns and len(pred_df) else 0,
        "forecast_min_utc": forecast_min.isoformat() if forecast_min is not None else None,
        "forecast_max_utc": forecast_max.isoformat() if forecast_max is not None else None,
        "current_time_utc": now_utc.isoformat(),
        "evaluation_available_at_utc": evaluation_available_at.isoformat() if evaluation_available_at is not None else None,
        "time_until_evaluation_seconds": wait_seconds,
        "time_until_evaluation_human": human_wait,
    }


def prepare_evaluation_frame(pred_df: pd.DataFrame, actual_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")

    pred_df = pred_df[pred_df["split_name"] == split_name].copy()
    if pred_df.empty:
        raise RuntimeError(f"No predictions found for split={split_name!r}")

    if "prediction_ready_utc" in pred_df.columns:
        latest_run_ts = pred_df["prediction_ready_utc"].max()
        pred_df = pred_df[pred_df["prediction_ready_utc"] == latest_run_ts].copy()

    mature_pred = pred_df[pred_df["forecast_timestamp_utc"] <= now_utc].copy()
    if mature_pred.empty:
        raise RuntimeError("No matured predictions yet; forecast horizon has not elapsed for this split.")

    if "target_column" not in mature_pred.columns:
        raise RuntimeError("Predictions do not have a target_column field.")

    target_column = str(mature_pred["target_column"].iloc[0])
    actual_df = actual_df[actual_df["split_name"] == split_name].copy()

    merged = resolve_actual_values(mature_pred, actual_df, target_column)

    matched = merged["actual_value"].notna().sum()
    if matched == 0:
        raise RuntimeError(
            f"No actuals matched for target_column={target_column!r} and split={split_name!r}. "
            "Check that dataset_splits_gauge_24h has non-null values for this column at the forecast timestamps."
        )

    merged["actual_available"] = merged["actual_value"].notna()
    merged = merged[merged["actual_available"]].copy()

    merged["error"] = merged["actual_value"] - merged["prediction"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2
    merged["ape"] = np.where(
        merged["actual_value"].abs() > 1e-9,
        merged["abs_error"] / merged["actual_value"].abs(),
        np.nan,
    )
    merged["threshold"] = merged["station_name"].map(THRESHOLD_BY_STATION)
    merged["actual_event_low_water"] = np.where(
        merged["threshold"].notna(),
        merged["actual_value"] <= merged["threshold"],
        pd.NA,
    )
    merged["pred_event_low_water"] = np.where(
        merged["threshold"].notna(),
        merged["prediction"] <= merged["threshold"],
        pd.NA,
    )
    return merged.sort_values(["forecast_timestamp_utc", "station_name"]).reset_index(drop=True)


def regression_metrics(df: pd.DataFrame) -> dict:
    mse = float(df["squared_error"].mean())
    rmse = float(np.sqrt(mse))
    mae = float(df["abs_error"].mean())
    bias = float(df["error"].mean())
    mape = float(df["ape"].dropna().mean()) if df["ape"].notna().any() else None
    p90_abs_error = float(df["abs_error"].quantile(0.90))
    return {
        "rows_evaluated": int(len(df)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "p90_abs_error": p90_abs_error,
    }


def classification_metrics(df: pd.DataFrame) -> dict:
    event_df = df.dropna(subset=["actual_event_low_water", "pred_event_low_water"]).copy()
    if event_df.empty:
        return {
            "event_rows": 0,
            "event_precision": None,
            "event_recall": None,
            "event_f1": None,
        }

    actual_event = event_df["actual_event_low_water"].astype(bool)
    pred_event = event_df["pred_event_low_water"].astype(bool)
    tp = int((actual_event & pred_event).sum())
    fp = int((~actual_event & pred_event).sum())
    fn = int((actual_event & ~pred_event).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "event_rows": int(len(event_df)),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def build_station_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_name, g in eval_df.groupby("station_name", dropna=False):
        row = {"station_name": station_name}
        row.update(regression_metrics(g))
        row.update(classification_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def build_model_metrics(eval_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    group_cols = ["model_version"]
    if "run_id" in eval_df.columns:
        group_cols.append("run_id")
    if "target_mode" in eval_df.columns:
        group_cols.append("target_mode")
    if "prediction_horizon_hours" in eval_df.columns:
        group_cols.append("prediction_horizon_hours")
    if "target_column" in eval_df.columns:
        group_cols.append("target_column")

    rows = []
    for keys, g in eval_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"split_name": split_name}
        row.update({col: val for col, val in zip(group_cols, keys)})
        row.update(regression_metrics(g))
        row.update(classification_metrics(g))
        row["forecast_min_utc"] = g["forecast_timestamp_utc"].min().isoformat()
        row["forecast_max_utc"] = g["forecast_timestamp_utc"].max().isoformat()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("forecast_max_utc").reset_index(drop=True)


def main():
    split_name = EVAL_SPLIT_NAME
    pred_df = load_predictions()
    actual_df = load_actuals()

    try:
        eval_df = prepare_evaluation_frame(pred_df, actual_df, split_name)
    except RuntimeError as e:
        split_pred_df = pred_df[pred_df["split_name"] == split_name].copy()
        target_column = str(split_pred_df["target_column"].iloc[0]) if "target_column" in split_pred_df.columns else None
        summary = build_not_ready_summary(split_pred_df, str(e), split_name, target_column)
        with open(EVALUATION_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("=== Gauge 24h evaluation not ready ===")
        print(json.dumps(summary, indent=2))
        print(f"\nWrote: {EVALUATION_SUMMARY_JSON}")
        sys.exit(0)

    overall_reg = regression_metrics(eval_df)
    overall_cls = classification_metrics(eval_df)

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "split_name": split_name,
        "source_predictions_table": PREDICTIONS_TABLE,
        "source_actuals_table": "dataset_splits_gauge_24h",
        **overall_reg,
        **overall_cls,
        "stations_evaluated": int(eval_df["station_name"].nunique()),
        "model_versions_evaluated": int(eval_df["model_version"].nunique()) if "model_version" in eval_df.columns else None,
        "target_modes_evaluated": sorted(eval_df["target_mode"].dropna().astype(str).unique().tolist()) if "target_mode" in eval_df.columns else None,
        "horizons_evaluated": sorted(pd.to_numeric(eval_df["prediction_horizon_hours"], errors="coerce").dropna().astype(int).unique().tolist()) if "prediction_horizon_hours" in eval_df.columns else None,
        "forecast_min_utc": eval_df["forecast_timestamp_utc"].min().isoformat(),
        "forecast_max_utc": eval_df["forecast_timestamp_utc"].max().isoformat(),
    }

    station_metrics_df = build_station_metrics(eval_df)
    model_metrics_df = build_model_metrics(eval_df, split_name)

    eval_df.to_csv(EVALUATION_CSV, index=False)
    station_metrics_df.to_csv(STATION_METRICS_CSV, index=False)
    model_metrics_df.to_csv(MODEL_METRICS_CSV, index=False)

    write_bigquery_table(eval_df, EVALUATION_TABLE, dataset="rhein_curated", if_exists="replace")

    with open(EVALUATION_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Gauge 24h evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print("\nWorst stations by RMSE:")
    print(station_metrics_df.head(10).to_string(index=False))
    print(f"\nWrote: {EVALUATION_CSV}")
    print(f"Wrote: {STATION_METRICS_CSV}")
    print(f"Wrote: {MODEL_METRICS_CSV}")
    print(f"Wrote BigQuery table: rhein_curated.{EVALUATION_TABLE}")
    print(f"Wrote: {EVALUATION_SUMMARY_JSON}")


if __name__ == "__main__":
    main()