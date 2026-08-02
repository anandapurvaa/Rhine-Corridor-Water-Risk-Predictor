from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from modeling.data_loader import load_bigquery_table, write_bigquery_table
from modeling.schemas import TABLE_NAME, TARGET_COLUMN

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_TABLE = "gauge_24h_production_predictions"
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


def load_predictions() -> pd.DataFrame:
    df = load_bigquery_table(
        PREDICTIONS_TABLE,
        dataset="rhein_curated",
        order_by="forecast_timestamp_utc, station_name",
    )

    if df.empty:
        raise ValueError(f"No rows found in rhein_curated.{PREDICTIONS_TABLE}")

    for col in ["timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "prediction" not in df.columns:
        raise ValueError("Predictions table must include prediction column")

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    return df


def load_actuals() -> pd.DataFrame:
    df = load_bigquery_table(
        TABLE_NAME,
        dataset="rhein_curated",
        columns=["station_name", "timestamp_utc", TARGET_COLUMN],
        order_by="timestamp_utc, station_name",
    )

    if df.empty:
        raise ValueError(f"No rows found in rhein_curated.{TABLE_NAME}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    return df


def prepare_evaluation_frame(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")

    matured_pred = pred_df[pred_df["forecast_timestamp_utc"] <= now_utc].copy()
    if matured_pred.empty:
        raise ValueError("No matured predictions yet; forecast horizon has not elapsed for any rows.")

    actual_df = actual_df.rename(
        columns={
            "timestamp_utc": "actual_timestamp_utc",
            TARGET_COLUMN: "actual_value",
        }
    )

    merged = matured_pred.merge(
        actual_df,
        left_on=["station_name", "forecast_timestamp_utc"],
        right_on=["station_name", "actual_timestamp_utc"],
        how="left",
        validate="many_to_one",
    )

    merged["actual_available"] = merged["actual_value"].notna()
    merged = merged[merged["actual_available"]].copy()

    if merged.empty:
        raise ValueError("No matured predictions have matching actuals yet.")

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


def build_model_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_version"]
    if "run_id" in eval_df.columns:
        group_cols.append("run_id")

    rows = []
    for keys, g in eval_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(regression_metrics(g))
        row.update(classification_metrics(g))
        row["forecast_min_utc"] = g["forecast_timestamp_utc"].min().isoformat()
        row["forecast_max_utc"] = g["forecast_timestamp_utc"].max().isoformat()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("forecast_max_utc").reset_index(drop=True)


def main():
    pred_df = load_predictions()
    actual_df = load_actuals()
    eval_df = prepare_evaluation_frame(pred_df, actual_df)

    overall_reg = regression_metrics(eval_df)
    overall_cls = classification_metrics(eval_df)

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions_table": PREDICTIONS_TABLE,
        "source_actuals_table": TABLE_NAME,
        **overall_reg,
        **overall_cls,
        "stations_evaluated": int(eval_df["station_name"].nunique()),
        "model_versions_evaluated": int(eval_df["model_version"].nunique()) if "model_version" in eval_df.columns else None,
        "forecast_min_utc": eval_df["forecast_timestamp_utc"].min().isoformat(),
        "forecast_max_utc": eval_df["forecast_timestamp_utc"].max().isoformat(),
    }

    station_metrics_df = build_station_metrics(eval_df)
    model_metrics_df = build_model_metrics(eval_df)

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