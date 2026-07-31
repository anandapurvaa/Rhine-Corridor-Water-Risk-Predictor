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

MIN_TRAIN_ROWS_PER_STATION = 80
TOP_N_BAD_STATIONS = 12


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


def build_station_residual_pipeline() -> Pipeline:
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
                ["timeseries_name", "source"],
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                [
                    "base_prediction",
                    "target_value",
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
                    "temperature_c",
                    "precipitation_mm",
                    "wind_speed_ms",
                    "pressure_hpa",
                    "relative_humidity_pct",
                    "distance_km",
                    "temp_change_1_3",
                    "pressure_delta_3",
                ],
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.03,
        max_iter=160,
        min_samples_leaf=20,
        l2_regularization=0.2,
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


def train_station_models(meta_train: pd.DataFrame, bad_stations: list[str]) -> dict:
    models = {}
    for station_name in bad_stations:
        station_df = meta_train[meta_train["station_name"] == station_name].copy()
        if len(station_df) < MIN_TRAIN_ROWS_PER_STATION:
            continue

        station_df["base_residual"] = station_df[TARGET_COLUMN] - station_df["base_prediction"]
        X_station = station_df.drop(columns=["base_residual", "station_name", "timestamp_utc"])
        y_station = station_df["base_residual"]

        model = build_station_residual_pipeline()
        model.fit(X_station, y_station)
        models[station_name] = model

    return models


def apply_station_models(meta_test: pd.DataFrame, station_models: dict) -> np.ndarray:
    adjustments = np.zeros(len(meta_test), dtype=float)

    for idx, row in meta_test.iterrows():
        station_name = row["station_name"]
        if station_name not in station_models:
            continue

        row_df = pd.DataFrame([row.drop(labels=["station_name", "timestamp_utc"])])
        adjustments[meta_test.index.get_loc(idx)] = station_models[station_name].predict(row_df)[0]

    return adjustments


def main():
    df = prepare_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    fold_predictions = []
    bad_station_log = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        meta_cols = ["timestamp_utc", "station_name", "timeseries_name", "source"] + NUMERIC_COLUMNS + [TARGET_COLUMN]
        meta_train = df.iloc[train_idx][meta_cols].copy()
        meta_test = df.iloc[test_idx][meta_cols].copy()

        global_model = build_global_pipeline()
        global_model.fit(X_train, y_train)

        base_pred_train = global_model.predict(X_train)
        base_pred_test = global_model.predict(X_test)

        meta_train["base_prediction"] = base_pred_train
        meta_test["base_prediction"] = base_pred_test

        train_eval = meta_train[["station_name", TARGET_COLUMN, "base_prediction"]].copy()
        train_eval["abs_error"] = (train_eval[TARGET_COLUMN] - train_eval["base_prediction"]).abs()

        station_rank = (
            train_eval.groupby("station_name", dropna=False)
            .agg(rows=("abs_error", "size"), mean_abs_error=("abs_error", "mean"))
            .reset_index()
            .query("rows >= @MIN_TRAIN_ROWS_PER_STATION")
            .sort_values("mean_abs_error", ascending=False)
        )

        bad_stations = station_rank.head(TOP_N_BAD_STATIONS)["station_name"].astype(str).tolist()
        station_models = train_station_models(meta_train, bad_stations)
        residual_adjustments = apply_station_models(meta_test, station_models)

        final_pred_test = base_pred_test + residual_adjustments

        metrics = regression_metrics(y_test, final_pred_test)
        metrics["fold"] = fold_num
        metrics["train_rows"] = int(len(train_idx))
        metrics["test_rows"] = int(len(test_idx))
        metrics["train_start"] = str(df.iloc[train_idx]["timestamp_utc"].min())
        metrics["train_end"] = str(df.iloc[train_idx]["timestamp_utc"].max())
        metrics["test_start"] = str(df.iloc[test_idx]["timestamp_utc"].min())
        metrics["test_end"] = str(df.iloc[test_idx]["timestamp_utc"].max())
        metrics["targeted_station_models"] = int(len(station_models))
        fold_metrics.append(metrics)

        bad_station_log.extend(
            [{"fold": fold_num, "station_name": s} for s in station_models.keys()]
        )

        fold_df = meta_test[["station_name", "timestamp_utc"]].copy()
        fold_df[TARGET_COLUMN] = y_test.values
        fold_df["base_prediction"] = base_pred_test
        fold_df["prediction"] = final_pred_test
        fold_df["predicted_residual_adjustment"] = residual_adjustments
        fold_df["fold"] = fold_num
        fold_predictions.append(add_error_columns(fold_df))

    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    station_metrics_df = station_level_metrics(predictions_df)
    bad_station_log_df = pd.DataFrame(bad_station_log)

    full_global_model = build_global_pipeline()
    full_global_model.fit(X, y)

    summary = {
        "mean_mae": float(fold_metrics_df["mae"].mean()),
        "mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "std_rmse": float(fold_metrics_df["rmse"].std(ddof=0)),
        "mean_r2": float(fold_metrics_df["r2"].mean()),
        "max_fold_rmse": float(fold_metrics_df["rmse"].max()),
        "rows_scored": int(len(predictions_df)),
    }

    joblib.dump(full_global_model, OUTPUT_DIR / "gauge_24h_targeted_station_residual_global_model.joblib")

    with open(OUTPUT_DIR / "gauge_24h_targeted_station_residual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fold_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_targeted_station_residual_fold_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "gauge_24h_targeted_station_residual_predictions.csv", index=False)
    station_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_targeted_station_residual_station_metrics.csv", index=False)
    bad_station_log_df.to_csv(OUTPUT_DIR / "gauge_24h_targeted_station_residual_station_log.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()