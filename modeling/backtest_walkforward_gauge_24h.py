from __future__ import annotations

from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.baselines_gauge_24h import (
    evaluate_regression,
    persistence_baseline,
    rolling_mean_baseline,
)
from modeling.data_loader import load_bigquery_table
from modeling.schemas import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS_LEAN,
    OPTIONAL_METADATA_COLUMNS,
    PRODUCTION_FEATURE_COLUMNS,
    TABLE_NAME,
    TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_DAYS = int(os.getenv("GAUGE24H_MIN_TRAIN_DAYS", "365"))
STEP_DAYS = int(os.getenv("GAUGE24H_STEP_DAYS", "30"))
HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
MAX_BACKTEST_MONTHS = int(os.getenv("GAUGE24H_MAX_BACKTEST_MONTHS", "18"))
MAX_STATIONS = int(os.getenv("GAUGE24H_MAX_STATIONS", "0"))


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_COLUMNS_LEAN,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_dataframe() -> pd.DataFrame:
    requested_columns = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS + OPTIONAL_METADATA_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]
        )
    )
    df = load_bigquery_table(
        TABLE_NAME,
        columns=requested_columns,
        order_by="timestamp_utc, station_name",
    )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS_LEAN + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)

    if MAX_BACKTEST_MONTHS > 0:
        cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=MAX_BACKTEST_MONTHS)
        df = df[df["timestamp_utc"] >= cutoff].copy()

    if MAX_STATIONS > 0:
        top_stations = (
            df.groupby("station_name", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(MAX_STATIONS)
            .index
        )
        df = df[df["station_name"].isin(top_stations)].copy()

    return df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)


def iter_walkforward_splits(df: pd.DataFrame):
    timestamps = pd.Series(sorted(pd.to_datetime(df["timestamp_utc"], utc=True).dropna().unique()))
    if timestamps.empty:
        return

    first_ts = pd.Timestamp(timestamps.min())
    last_ts = pd.Timestamp(timestamps.max())

    origin = first_ts + pd.Timedelta(f"{MIN_TRAIN_DAYS}D")
    step_delta = pd.Timedelta(f"{STEP_DAYS}D")
    horizon_delta = pd.Timedelta(f"{HORIZON_HOURS}h")

    while origin + horizon_delta <= last_ts:
        train_mask = df["timestamp_utc"] < origin
        test_mask = (df["timestamp_utc"] >= origin) & (df["timestamp_utc"] < origin + step_delta)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            yield origin, train_df, test_df

        origin = origin + step_delta


def add_event_metrics(frame: pd.DataFrame, threshold_by_station: dict[str, float], pred_col: str) -> dict:
    df = frame.copy()
    df["threshold"] = df["station_name"].map(threshold_by_station)
    df = df.dropna(subset=["threshold", TARGET_COLUMN, pred_col]).copy()

    if df.empty:
        return {
            "event_precision": None,
            "event_recall": None,
            "event_f1": None,
        }

    actual_event = df[TARGET_COLUMN] <= df["threshold"]
    pred_event = df[pred_col] <= df["threshold"]

    tp = int((actual_event & pred_event).sum())
    fp = int((~actual_event & pred_event).sum())
    fn = int((actual_event & ~pred_event).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def main():
    threshold_by_station = {
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

    df = prepare_dataframe()

    fold_rows = []
    pred_rows = []

    for fold_num, (origin, train_df, test_df) in enumerate(iter_walkforward_splits(df), start=1):
        X_train = train_df[PRODUCTION_FEATURE_COLUMNS].copy()
        y_train = train_df[TARGET_COLUMN].copy()
        X_test = test_df[PRODUCTION_FEATURE_COLUMNS].copy()
        y_test = test_df[TARGET_COLUMN].copy()

        model = build_pipeline()
        model.fit(X_train, y_train)

        pred_model = model.predict(X_test)
        pred_persist = persistence_baseline(test_df)
        pred_roll = rolling_mean_baseline(test_df)

        model_metrics = evaluate_regression(y_test, pred_model)
        persist_metrics = evaluate_regression(y_test, pred_persist)
        roll_metrics = evaluate_regression(y_test, pred_roll)

        fold_pred = test_df[["station_name", "timestamp_utc"]].copy()
        fold_pred[TARGET_COLUMN] = y_test.values
        fold_pred["prediction_model"] = pred_model
        fold_pred["prediction_persistence"] = pred_persist
        fold_pred["prediction_rolling_mean"] = pred_roll
        fold_pred["fold"] = fold_num
        fold_pred["forecast_origin_utc"] = origin
        pred_rows.append(fold_pred)

        event_metrics_model = add_event_metrics(fold_pred, threshold_by_station, "prediction_model")

        fold_rows.append(
            {
                "fold": fold_num,
                "forecast_origin_utc": str(origin),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_start": str(train_df["timestamp_utc"].min()),
                "train_end": str(train_df["timestamp_utc"].max()),
                "test_start": str(test_df["timestamp_utc"].min()),
                "test_end": str(test_df["timestamp_utc"].max()),
                "model_mae": model_metrics["mae"],
                "model_rmse": model_metrics["rmse"],
                "model_bias": model_metrics["bias"],
                "persist_mae": persist_metrics["mae"],
                "persist_rmse": persist_metrics["rmse"],
                "roll_mae": roll_metrics["mae"],
                "roll_rmse": roll_metrics["rmse"],
                **event_metrics_model,
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    if folds_df.empty:
        raise ValueError("No walk-forward folds were created. Check date coverage and MIN_TRAIN_DAYS.")

    summary = {
        "folds": int(len(folds_df)),
        "rows_scored": int(len(preds_df)),
        "min_train_days": MIN_TRAIN_DAYS,
        "step_days": STEP_DAYS,
        "max_backtest_months": MAX_BACKTEST_MONTHS,
        "max_stations": MAX_STATIONS,
        "mean_model_mae": float(folds_df["model_mae"].mean()),
        "mean_model_rmse": float(folds_df["model_rmse"].mean()),
        "mean_persist_rmse": float(folds_df["persist_rmse"].mean()),
        "mean_roll_rmse": float(folds_df["roll_rmse"].mean()),
        "model_vs_persist_rmse_gain": float(folds_df["persist_rmse"].mean() - folds_df["model_rmse"].mean()),
        "model_vs_roll_rmse_gain": float(folds_df["roll_rmse"].mean() - folds_df["model_rmse"].mean()),
        "mean_event_precision": float(folds_df["event_precision"].fillna(0).mean()),
        "mean_event_recall": float(folds_df["event_recall"].fillna(0).mean()),
        "mean_event_f1": float(folds_df["event_f1"].fillna(0).mean()),
    }

    folds_df.to_csv(OUTPUT_DIR / "gauge_24h_walkforward_fold_metrics.csv", index=False)
    preds_df.to_csv(OUTPUT_DIR / "gauge_24h_walkforward_predictions.csv", index=False)

    with open(OUTPUT_DIR / "gauge_24h_walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    final_model = build_pipeline()
    final_model.fit(df[PRODUCTION_FEATURE_COLUMNS], df[TARGET_COLUMN])
    joblib.dump(final_model, OUTPUT_DIR / "gauge_24h_walkforward_last_model.joblib")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()