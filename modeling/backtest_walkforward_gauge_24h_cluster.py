from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.baselines_gauge_24h import evaluate_regression, persistence_baseline
from modeling.data_loader import load_bigquery_table
from modeling.schemas import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS_LEAN,
    OPTIONAL_METADATA_COLUMNS,
    PRODUCTION_FEATURE_COLUMNS,
    ROBUSTNESS_REQUIRED_COLUMNS,
    BACKTEST_TABLE_NAME,
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_PLAN_PATH = OUTPUT_DIR / "station_cluster_plan_v2.csv"

MIN_TRAIN_DAYS = int(os.getenv("GAUGE24H_MIN_TRAIN_DAYS", "365"))
STEP_DAYS = int(os.getenv("GAUGE24H_STEP_DAYS", "30"))
HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
MAX_BACKTEST_MONTHS = int(os.getenv("GAUGE24H_MAX_BACKTEST_MONTHS", "18"))
MAX_STATIONS = int(os.getenv("GAUGE24H_MAX_STATIONS", "0"))
MIN_CLUSTER_TRAIN_ROWS = int(os.getenv("GAUGE24H_MIN_CLUSTER_TRAIN_ROWS", "500"))
MIN_CLUSTER_STATIONS = int(os.getenv("GAUGE24H_MIN_CLUSTER_STATIONS", "2"))
TARGET_MODE = os.getenv("GAUGE24H_TARGET_MODE", "level").strip().lower()
FEATURE_SET = os.getenv("GAUGE24H_FEATURE_SET", "default").strip().lower()
MAX_FEATURE_NULL_FRACTION = float(os.getenv("GAUGE24H_MAX_FEATURE_NULL_FRACTION", "0.35"))

BASELINE_FALLBACK_COLUMNS = ["target_value", "lag_1", "lag_3", "lag_6", "rolling_mean_3"]

def resolve_feature_columns(available_columns: list[str]) -> list[str]:
    base = [c for c in PRODUCTION_FEATURE_COLUMNS if c in available_columns]
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

def build_pipeline(feature_columns: list[str]) -> Pipeline:
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric_features = [c for c in feature_columns if c not in categorical_features]

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

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
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

def load_cluster_plan() -> pd.DataFrame:
    if not CLUSTER_PLAN_PATH.exists():
        raise FileNotFoundError(f"Missing cluster plan: {CLUSTER_PLAN_PATH}")

    cluster_df = pd.read_csv(CLUSTER_PLAN_PATH)
    cluster_cols = [c for c in cluster_df.columns if c.startswith("cluster")]
    if "station_name" not in cluster_df.columns or not cluster_cols:
        raise ValueError("Cluster plan must include station_name and a cluster column")

    cluster_col = cluster_cols[0]
    cluster_df = cluster_df[["station_name", cluster_col]].copy()
    cluster_df = cluster_df.rename(columns={cluster_col: "cluster"})
    cluster_df["station_name"] = cluster_df["station_name"].astype("string")
    cluster_df["cluster"] = pd.to_numeric(cluster_df["cluster"], errors="coerce").astype("Int64")
    cluster_df = cluster_df.drop_duplicates(subset=["station_name"]).reset_index(drop=True)
    return cluster_df

def prepare_dataframe() -> tuple[pd.DataFrame, str, list[str]]:
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
        BACKTEST_TABLE_NAME,
        columns=requested_columns,
        order_by="timestamp_utc, station_name",
        allow_missing_columns=True,
    )

    missing_required = [c for c in ROBUSTNESS_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required base columns: {missing_required}")

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

    feature_columns = resolve_feature_columns(df.columns.tolist())
    if not feature_columns:
        raise ValueError("No usable feature columns found for cluster backtest.")

    row_null_fraction = df[feature_columns].isna().mean(axis=1)
    df = df.loc[row_null_fraction <= MAX_FEATURE_NULL_FRACTION].copy()

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

    return df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True), target_column, feature_columns

def iter_walkforward_splits(df: pd.DataFrame):
    timestamps = pd.Series(sorted(pd.to_datetime(df["timestamp_utc"], utc=True).dropna().unique()))
    if timestamps.empty:
        return

    first_ts = pd.Timestamp(timestamps.min())
    last_ts = pd.Timestamp(timestamps.max())

    origin = first_ts + pd.to_timedelta(MIN_TRAIN_DAYS, unit="D")
    step_delta = pd.to_timedelta(STEP_DAYS, unit="D")
    horizon_delta = pd.to_timedelta(HORIZON_HOURS, unit="h")

    # Capping training data strictly to the end of 2025
    max_train_ts = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")

    while origin + horizon_delta <= last_ts:
        train_mask = (df["timestamp_utc"] < origin) & (df["timestamp_utc"] <= max_train_ts)
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

def summarize_cluster_coverage(train_df: pd.DataFrame, cluster_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = train_df.merge(cluster_df, on="station_name", how="inner")
    coverage = (
        merged.groupby("cluster", dropna=False)
        .agg(
            rows=("station_name", "size"),
            stations=("station_name", "nunique"),
        )
        .reset_index()
    )
    coverage["eligible"] = (
        (coverage["rows"] >= MIN_CLUSTER_TRAIN_ROWS)
        & (coverage["stations"] >= MIN_CLUSTER_STATIONS)
    )

    cluster_plan_counts = (
        cluster_df.groupby("cluster", dropna=False)["station_name"]
        .nunique()
        .rename("planned_stations")
        .reset_index()
    )
    coverage = coverage.merge(cluster_plan_counts, on="cluster", how="outer")
    coverage["rows"] = coverage["rows"].fillna(0).astype(int)
    coverage["stations"] = coverage["stations"].fillna(0).astype(int)
    coverage["planned_stations"] = coverage["planned_stations"].fillna(0).astype(int)
    coverage["eligible"] = coverage["eligible"].fillna(False)

    summary = {
        "planned_clusters": int(cluster_df["cluster"].dropna().nunique()),
        "matched_clusters": int(coverage.loc[coverage["rows"] > 0, "cluster"].nunique()),
        "eligible_clusters": int(coverage["eligible"].sum()),
        "unmatched_clusters": int((coverage["rows"] == 0).sum()),
    }
    return coverage.sort_values("cluster").reset_index(drop=True), summary

def fit_cluster_models(
    train_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[dict[int, Pipeline], pd.DataFrame]:
    merged = train_df.merge(cluster_df, on="station_name", how="inner")
    coverage, _ = summarize_cluster_coverage(train_df, cluster_df)
    models: dict[int, Pipeline] = {}

    for row in coverage.itertuples(index=False):
        cluster_id = row.cluster
        if pd.isna(cluster_id) or not row.eligible:
            continue

        cluster_train = merged[merged["cluster"] == cluster_id].copy()
        if cluster_train.empty:
            continue

        model = build_pipeline(feature_columns)
        model.fit(cluster_train[feature_columns], build_targets(cluster_train, target_column))
        models[int(cluster_id)] = model

    return models, coverage

def apply_cluster_models(
    test_df: pd.DataFrame,
    base_pred: np.ndarray,
    cluster_df: pd.DataFrame,
    cluster_models: dict[int, Pipeline],
    target_column: str,
    feature_columns: list[str],
):
    pred = base_pred.copy()
    merged = test_df[["station_name"]].merge(cluster_df, on="station_name", how="left")

    model_source = np.array(["global_fallback"] * len(test_df), dtype=object)

    for cluster_id, model in cluster_models.items():
        mask = (merged["cluster"] == cluster_id).fillna(False).to_numpy()
        if mask.sum() == 0:
            continue
        pred_raw = model.predict(test_df.loc[mask, feature_columns])
        _, pred_eval = reconstruct_predictions(test_df.loc[mask], pred_raw, target_column)
        pred[mask] = pred_eval
        model_source[mask] = f"cluster_{cluster_id}"

    return pred, merged["cluster"], pd.Series(model_source, index=test_df.index, name="model_source")

def main():
    df, target_column, feature_columns = prepare_dataframe()
    cluster_df = load_cluster_plan()

    fold_rows = []
    pred_rows = []
    coverage_rows = []

    for fold_num, (origin, train_df, test_df) in enumerate(iter_walkforward_splits(df), start=1):
        global_model = build_pipeline(feature_columns)
        global_model.fit(train_df[feature_columns], build_targets(train_df, target_column))

        pred_global_raw = global_model.predict(test_df[feature_columns])
        y_test_eval, pred_global = reconstruct_predictions(test_df, pred_global_raw, target_column)
        pred_persist = persistence_baseline(test_df)

        cluster_models, coverage = fit_cluster_models(train_df, cluster_df, target_column, feature_columns)
        pred_cluster, cluster_series, model_source = apply_cluster_models(
            test_df=test_df,
            base_pred=pred_global.copy(),
            cluster_df=cluster_df,
            cluster_models=cluster_models,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        global_metrics = evaluate_regression(y_test_eval, pred_global)
        cluster_metrics = evaluate_regression(y_test_eval, pred_cluster)
        persist_metrics = evaluate_regression(y_test_eval, pred_persist)

        planned_clusters = int(cluster_df["cluster"].dropna().nunique())
        matched_clusters = int((coverage["rows"] > 0).sum())
        eligible_clusters = int(coverage["eligible"].sum())
        unmatched_clusters = int((coverage["rows"] == 0).sum())
        cluster_assigned_rows = int(model_source.str.startswith("cluster_").sum())
        global_fallback_rows = int((model_source == "global_fallback").sum())

        for row in coverage.itertuples(index=False):
            coverage_rows.append(
                {
                    "fold": fold_num,
                    "forecast_origin_utc": str(origin),
                    "cluster": None if pd.isna(row.cluster) else int(row.cluster),
                    "planned_stations": int(row.planned_stations),
                    "matched_train_rows": int(row.rows),
                    "matched_train_stations": int(row.stations),
                    "eligible": bool(row.eligible),
                    "model_trained": bool((not pd.isna(row.cluster)) and int(row.cluster) in cluster_models),
                }
            )

        fold_rows.append(
            {
                "fold": fold_num,
                "forecast_origin_utc": str(origin),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "cluster_models_used": int(len(cluster_models)),
                "planned_clusters": planned_clusters,
                "matched_clusters": matched_clusters,
                "eligible_clusters": eligible_clusters,
                "unmatched_clusters": unmatched_clusters,
                "cluster_assigned_rows": cluster_assigned_rows,
                "global_fallback_rows": global_fallback_rows,
                "target_column": target_column,
                "target_mode": TARGET_MODE,
                "feature_set": FEATURE_SET,
                "feature_count": int(len(feature_columns)),
                "feature_columns_used": ",".join(feature_columns),
                "horizon_hours": HORIZON_HOURS,
                "global_rmse": global_metrics["rmse"],
                "global_mae": global_metrics["mae"],
                "cluster_rmse": cluster_metrics["rmse"],
                "cluster_mae": cluster_metrics["mae"],
                "persist_rmse": persist_metrics["rmse"],
                "persist_mae": persist_metrics["mae"],
                "cluster_vs_global_rmse_gain": float(global_metrics["rmse"] - cluster_metrics["rmse"]),
                "cluster_vs_persist_rmse_gain": float(persist_metrics["rmse"] - cluster_metrics["rmse"]),
            }
        )

        fold_pred = test_df[["station_name", "timestamp_utc"]].copy()
        fold_pred[target_column] = y_test_eval.values
        fold_pred["prediction_global"] = pred_global
        fold_pred["prediction_cluster"] = pred_cluster
        fold_pred["prediction_persistence"] = pred_persist
        fold_pred["cluster"] = cluster_series.values
        fold_pred["model_source"] = model_source.values
        fold_pred["fold"] = fold_num
        fold_pred["forecast_origin_utc"] = origin
        pred_rows.append(fold_pred)

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows)

    if folds_df.empty:
        raise ValueError("No walk-forward folds were created.")

    summary = {
        "folds": int(len(folds_df)),
        "rows_scored": int(len(preds_df)),
        "min_train_days": MIN_TRAIN_DAYS,
        "step_days": STEP_DAYS,
        "horizon_hours": HORIZON_HOURS,
        "target_column": target_column,
        "target_mode": TARGET_MODE,
        "feature_set": FEATURE_SET,
        "feature_count": int(len(feature_columns)),
        "feature_columns_used": feature_columns,
        "max_backtest_months": MAX_BACKTEST_MONTHS,
        "max_stations": MAX_STATIONS,
        "min_cluster_train_rows": MIN_CLUSTER_TRAIN_ROWS,
        "min_cluster_stations": MIN_CLUSTER_STATIONS,
        "max_feature_null_fraction": MAX_FEATURE_NULL_FRACTION,
        "mean_global_rmse": float(folds_df["global_rmse"].mean()),
        "mean_cluster_rmse": float(folds_df["cluster_rmse"].mean()),
        "mean_persist_rmse": float(folds_df["persist_rmse"].mean()),
        "cluster_vs_global_rmse_gain": float(folds_df["cluster_vs_global_rmse_gain"].mean()),
        "cluster_vs_persist_rmse_gain": float(folds_df["cluster_vs_persist_rmse_gain"].mean()),
        "mean_cluster_models_used": float(folds_df["cluster_models_used"].mean()),
        "mean_planned_clusters": float(folds_df["planned_clusters"].mean()),
        "mean_matched_clusters": float(folds_df["matched_clusters"].mean()),
        "mean_eligible_clusters": float(folds_df["eligible_clusters"].mean()),
        "mean_unmatched_clusters": float(folds_df["unmatched_clusters"].mean()),
        "mean_cluster_assigned_rows": float(folds_df["cluster_assigned_rows"].mean()),
        "mean_global_fallback_rows": float(folds_df["global_fallback_rows"].mean()),
    }

    folds_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_benchmark_fold_metrics.csv", index=False)
    preds_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_benchmark_predictions.csv", index=False)
    coverage_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_benchmark_cluster_coverage.csv", index=False)

    with open(OUTPUT_DIR / "gauge_24h_cluster_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()