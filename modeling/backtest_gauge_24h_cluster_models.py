from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.data_loader import load_bigquery_table


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "supervised_gauge_24h_multisource"
TARGET_COLUMN = "target_value_t_plus_24h"
CLUSTER_PLAN_PATH = OUTPUT_DIR / "station_cluster_plan.csv"

CATEGORICAL_COLUMNS = [
    "station_name",
    "timeseries_name",
    "unit",
    "source",
]

NUMERIC_COLUMNS = [
    "target_value",
    "distance_km",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
    "lag_1",
    "lag_3",
    "lag_6",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_min_6",
    "rolling_max_6",
    "hour_utc",
    "day_of_week",
    "month",
    "temp_lag_6",
    "temp_lag_12",
    "precip_lag_6",
    "precip_lag_12",
    "wind_lag_6",
    "pressure_lag_3",
    "humidity_lag_1",
    "temp_roll_mean_6",
    "temp_roll_mean_12",
    "precip_roll_sum_12",
    "precip_roll_sum_24",
    "wind_roll_mean_12",
    "pressure_roll_mean_12",
    "humidity_roll_mean_12",
    "pressure_delta_3",
    "temp_change_1_3",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
MIN_CLUSTER_TRAIN_ROWS = 120


def build_global_pipeline() -> Pipeline:
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
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=350,
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


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    residuals = y_true - y_pred
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_residual": float(np.mean(residuals)),
        "median_abs_error": float(np.median(np.abs(residuals))),
        "p90_abs_error": float(np.percentile(np.abs(residuals), 90)),
        "max_abs_error": float(np.max(np.abs(residuals))),
    }


def prepare_dataframe() -> pd.DataFrame:
    df = load_bigquery_table(TABLE_NAME)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df[TARGET_COLUMN].notna()].copy()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(["timestamp_utc", "station_name", "timeseries_name"]).reset_index(drop=True)
    return df


def load_cluster_plan():
    candidates = [
        OUTPUT_DIR / "station_cluster_plan_v2.csv",
        OUTPUT_DIR / "station_cluster_plan.csv",
    ]
    for path in candidates:
        if path.exists():
            cluster_df = pd.read_csv(path)
            if "station_name" not in cluster_df.columns:
                raise ValueError(f"{path.name} must include station_name")
            cluster_cols = [c for c in cluster_df.columns if c.startswith("cluster")]
            if not cluster_cols:
                raise ValueError(f"{path.name} must include a cluster column")
            cluster_col = cluster_cols[0]
            cluster_df = cluster_df[["station_name", cluster_col]].copy()
            cluster_df = cluster_df.rename(columns={cluster_col: "cluster"})
            cluster_df["station_name"] = cluster_df["station_name"].astype("string")
            cluster_df["cluster"] = cluster_df["cluster"].astype(int)
            return cluster_df
    raise FileNotFoundError("Missing station_cluster_plan_v2.csv or station_cluster_plan.csv")


def station_level_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_name, g in scored_df.groupby("station_name", dropna=False):
        metrics = regression_metrics(g[TARGET_COLUMN], g["prediction"])
        rows.append({"station_name": station_name, "rows": int(len(g)), **metrics})
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["residual"] = pred_df[TARGET_COLUMN] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    return pred_df


def fit_cluster_models(train_df: pd.DataFrame, cluster_plan: pd.DataFrame) -> dict:
    merged = train_df.merge(cluster_plan, on="station_name", how="inner")
    models = {}

    for cluster_id, cluster_train in merged.groupby("cluster"):
        if cluster_train["station_name"].nunique() < 2:
            continue
        if len(cluster_train) < MIN_CLUSTER_TRAIN_ROWS:
            continue

        X_cluster = cluster_train[FEATURE_COLUMNS].copy()
        y_cluster = cluster_train[TARGET_COLUMN].copy()

        model = build_global_pipeline()
        model.fit(X_cluster, y_cluster)
        models[int(cluster_id)] = model

    return models


def apply_cluster_models(test_df: pd.DataFrame, base_pred: np.ndarray, cluster_plan: pd.DataFrame, cluster_models: dict):
    out_pred = base_pred.copy()
    merged_test = test_df[["station_name"]].merge(cluster_plan, on="station_name", how="left")

    for cluster_id, model in cluster_models.items():
        mask = merged_test["cluster"] == cluster_id
        if mask.sum() == 0:
            continue
        out_pred[mask.to_numpy()] = model.predict(test_df.loc[mask.to_numpy(), FEATURE_COLUMNS])

    return out_pred, merged_test["cluster"]


def main():
    df = prepare_dataframe()
    cluster_plan = load_cluster_plan()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    fold_predictions = []
    cluster_usage_rows = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        X_train = train_df[FEATURE_COLUMNS].copy()
        X_test = test_df[FEATURE_COLUMNS].copy()
        y_train = train_df[TARGET_COLUMN].copy()
        y_test = test_df[TARGET_COLUMN].copy()

        global_model = build_global_pipeline()
        global_model.fit(X_train, y_train)
        base_pred_test = global_model.predict(X_test)

        cluster_models = fit_cluster_models(train_df, cluster_plan)
        final_pred_test, cluster_series = apply_cluster_models(test_df, base_pred_test, cluster_plan, cluster_models)

        metrics = regression_metrics(y_test, final_pred_test)
        metrics["fold"] = fold_num
        metrics["train_rows"] = int(len(train_idx))
        metrics["test_rows"] = int(len(test_idx))
        metrics["cluster_models_used"] = int(len(cluster_models))
        metrics["train_start"] = str(train_df["timestamp_utc"].min())
        metrics["train_end"] = str(train_df["timestamp_utc"].max())
        metrics["test_start"] = str(test_df["timestamp_utc"].min())
        metrics["test_end"] = str(test_df["timestamp_utc"].max())
        fold_metrics.append(metrics)

        for cluster_id in sorted(cluster_models.keys()):
            cluster_usage_rows.append({"fold": fold_num, "cluster": cluster_id})

        fold_df = test_df[["station_name", "timestamp_utc"]].copy()
        fold_df[TARGET_COLUMN] = y_test.values
        fold_df["base_prediction"] = base_pred_test
        fold_df["prediction"] = final_pred_test
        fold_df["cluster"] = cluster_series.values
        fold_df["fold"] = fold_num
        fold_predictions.append(add_error_columns(fold_df))

    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    station_metrics_df = station_level_metrics(predictions_df)
    cluster_usage_df = pd.DataFrame(cluster_usage_rows)

    full_global_model = build_global_pipeline()
    full_global_model.fit(X, y)
    joblib.dump(full_global_model, OUTPUT_DIR / "gauge_24h_cluster_backbone_model.joblib")

    summary = {
        "mean_mae": float(fold_metrics_df["mae"].mean()),
        "mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "std_rmse": float(fold_metrics_df["rmse"].std(ddof=0)),
        "mean_r2": float(fold_metrics_df["r2"].mean()),
        "max_fold_rmse": float(fold_metrics_df["rmse"].max()),
        "rows_scored": int(len(predictions_df)),
    }

    with open(OUTPUT_DIR / "gauge_24h_cluster_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fold_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_models_fold_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_models_predictions.csv", index=False)
    station_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_models_station_metrics.csv", index=False)
    cluster_usage_df.to_csv(OUTPUT_DIR / "gauge_24h_cluster_models_cluster_usage.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()