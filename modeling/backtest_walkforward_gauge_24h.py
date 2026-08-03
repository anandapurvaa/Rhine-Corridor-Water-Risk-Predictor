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
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_DAYS = int(os.getenv("GAUGE24H_MIN_TRAIN_DAYS", "365"))
STEP_DAYS = int(os.getenv("GAUGE24H_STEP_DAYS", "30"))
HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
MAX_BACKTEST_MONTHS = int(os.getenv("GAUGE24H_MAX_BACKTEST_MONTHS", "18"))
MAX_STATIONS = int(os.getenv("GAUGE24H_MAX_STATIONS", "0"))
TARGET_MODE = os.getenv("GAUGE24H_TARGET_MODE", "level").strip().lower()
FEATURE_SET = os.getenv("GAUGE24H_FEATURE_SET", "default").strip().lower()

BASELINE_FALLBACK_COLUMNS = ["target_value", "lag_1", "lag_3", "lag_6", "rolling_mean_3"]


def resolve_feature_columns() -> list[str]:
    base = list(dict.fromkeys(PRODUCTION_FEATURE_COLUMNS))
    if FEATURE_SET == "default":
        return base
    if FEATURE_SET == "simple":
        preferred = [
            c for c in ["target_value", "lag_1", "lag_3", "lag_6", "rolling_mean_3", "station_name"]
            if c in base
        ]
        if preferred:
            return preferred
        fallback = [c for c in base if c in BASELINE_FALLBACK_COLUMNS]
        return fallback or base
    raise ValueError(f"Unsupported GAUGE24H_FEATURE_SET={FEATURE_SET!r}")


FEATURE_COLUMNS = resolve_feature_columns()


def build_pipeline() -> Pipeline:
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in FEATURE_COLUMNS]
    numeric_features = [c for c in FEATURE_COLUMNS if c not in categorical_features]

    transformers = []
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def infer_step_hours(df: pd.DataFrame) -> float:
    diffs = (
        df.sort_values(["station_name", "timestamp_utc"])
        .groupby("station_name")["timestamp_utc"]
        .diff()
        .dropna()
    )
    if diffs.empty:
        raise ValueError("Cannot infer sampling frequency: no timestamp diffs found.")
    median_step = diffs.median()
    return median_step.total_seconds() / 3600.0


def build_horizon_target_column(df: pd.DataFrame, horizon_hours: int) -> tuple[pd.DataFrame, str]:
    native_col = f"target_value_t_plus_{horizon_hours}h"
    if native_col in df.columns:
        return df, native_col

    step_hours = infer_step_hours(df)
    if step_hours <= 0:
        raise ValueError(f"Invalid inferred step size: {step_hours} hours")

    shift_steps = round(horizon_hours / step_hours)
    if shift_steps <= 0:
        raise ValueError(
            f"Horizon {horizon_hours}h is smaller than sampling step {step_hours}h; cannot build target."
        )

    df = df.sort_values(["station_name", "timestamp_utc"]).copy()
    df[native_col] = df.groupby("station_name")["target_value"].shift(-shift_steps)
    return df, native_col


def prepare_dataframe() -> tuple[pd.DataFrame, str]:
    requested_columns = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS
            + OPTIONAL_METADATA_COLUMNS
            + [
                DEFAULT_TARGET_COLUMN,
                "timestamp_utc",
                "target_value",
                "lag_1",
                "lag_3",
                "lag_6",
                "rolling_mean_3",
            ]
        )
    )
    df = load_bigquery_table(
        TABLE_NAME,
        columns=requested_columns,
        order_by="timestamp_utc, station_name",
    )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    for col in [c for c in CATEGORICAL_COLUMNS if c in df.columns]:
        df[col] = df[col].astype("string")

    numeric_cols = sorted(
        set(NUMERIC_COLUMNS_LEAN + [DEFAULT_TARGET_COLUMN, "target_value", "lag_1", "lag_3", "lag_6", "rolling_mean_3"])
        & set(df.columns)
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_value"]).copy()
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)

    df, target_column = build_horizon_target_column(df, HORIZON_HOURS)
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column]).copy()
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

    return df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True), target_column


def iter_walkforward_splits(df: pd.DataFrame):
    timestamps = pd.Series(sorted(pd.to_datetime(df["timestamp_utc"], utc=True).dropna().unique()))
    if timestamps.empty:
        return

    first_ts = pd.Timestamp(timestamps.min())
    last_ts = pd.Timestamp(timestamps.max())

    origin = first_ts + pd.Timedelta(days=MIN_TRAIN_DAYS)
    step_delta = pd.Timedelta(days=STEP_DAYS)
    horizon_delta = pd.Timedelta(hours=HORIZON_HOURS)

    while origin + horizon_delta <= last_ts:
        train_mask = df["timestamp_utc"] < origin
        test_mask = (df["timestamp_utc"] >= origin) & (df["timestamp_utc"] < origin + step_delta)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            yield origin, train_df, test_df

        origin = origin + step_delta


def build_targets(frame: pd.DataFrame, target_column: str) -> pd.Series:
    y_level = pd.to_numeric(frame[target_column], errors="coerce")
    if TARGET_MODE == "delta":
        y_now = pd.to_numeric(frame["target_value"], errors="coerce")
        return y_level - y_now
    if TARGET_MODE == "level":
        return y_level
    raise ValueError(f"Unsupported GAUGE24H_TARGET_MODE={TARGET_MODE!r}")


def reconstruct_predictions(frame: pd.DataFrame, pred_raw: np.ndarray, target_column: str) -> tuple[pd.Series, np.ndarray]:
    y_true_eval = pd.to_numeric(frame[target_column], errors="coerce")
    if TARGET_MODE == "delta":
        y_now = pd.to_numeric(frame["target_value"], errors="coerce").to_numpy()
        return y_true_eval, pred_raw + y_now
    return y_true_eval, pred_raw


def add_event_metrics(frame: pd.DataFrame, threshold_by_station: dict[str, float], target_column: str, pred_col: str) -> dict:
    df = frame.copy()
    df["threshold"] = df["station_name"].map(threshold_by_station)
    df = df.dropna(subset=["threshold", target_column, pred_col]).copy()

    if df.empty:
        return {"event_precision": None, "event_recall": None, "event_f1": None}

    actual_event = df[target_column] <= df["threshold"]
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

    df, target_column = prepare_dataframe()

    fold_rows = []
    pred_rows = []

    for fold_num, (origin, train_df, test_df) in enumerate(iter_walkforward_splits(df), start=1):
        X_train = train_df[FEATURE_COLUMNS].copy()
        y_train = build_targets(train_df, target_column)
        X_test = test_df[FEATURE_COLUMNS].copy()

        model = build_pipeline()
        model.fit(X_train, y_train)

        pred_raw = model.predict(X_test)
        y_test_eval, pred_model = reconstruct_predictions(test_df, pred_raw, target_column)
        pred_persist = persistence_baseline(test_df)
        pred_roll = rolling_mean_baseline(test_df)

        model_metrics = evaluate_regression(y_test_eval, pred_model)
        persist_metrics = evaluate_regression(y_test_eval, pred_persist)
        roll_metrics = evaluate_regression(y_test_eval, pred_roll)

        fold_pred = test_df[["station_name", "timestamp_utc"]].copy()
        fold_pred[target_column] = y_test_eval.values
        fold_pred["prediction_model"] = pred_model
        fold_pred["prediction_persistence"] = pred_persist
        fold_pred["prediction_rolling_mean"] = pred_roll
        fold_pred["fold"] = fold_num
        fold_pred["forecast_origin_utc"] = origin
        pred_rows.append(fold_pred)

        event_metrics_model = add_event_metrics(fold_pred, threshold_by_station, target_column, "prediction_model")

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
                "target_column": target_column,
                "target_mode": TARGET_MODE,
                "feature_set": FEATURE_SET,
                "horizon_hours": HORIZON_HOURS,
                "feature_count": int(len(FEATURE_COLUMNS)),
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
        "horizon_hours": HORIZON_HOURS,
        "target_column": target_column,
        "target_mode": TARGET_MODE,
        "feature_set": FEATURE_SET,
        "feature_count": int(len(FEATURE_COLUMNS)),
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
    final_model.fit(df[FEATURE_COLUMNS], build_targets(df, target_column))
    joblib.dump(final_model, OUTPUT_DIR / "gauge_24h_walkforward_last_model.joblib")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()