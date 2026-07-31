from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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

BASE_NUMERIC_COLUMNS = [
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

REGIME_NUMERIC_COLUMNS = [
    "value_trend_3_12",
    "value_trend_6_24",
    "value_volatility_6",
    "value_volatility_12",
    "value_volatility_24",
    "value_vol_ratio_6_24",
    "value_range_6",
    "value_range_12",
    "value_accel_1_3",
    "value_accel_3_6",
    "weather_temp_trend_3_12",
    "weather_precip_intensity_6_24",
    "weather_wind_trend_3_12",
    "weather_pressure_trend_3_12",
]

NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + REGIME_NUMERIC_COLUMNS
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


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


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_keys = ["station_name", "timeseries_name"]

    df["value_trend_3_12"] = (df.groupby(group_keys)["target_value"].transform(lambda s: s.rolling(3, min_periods=2).mean() - s.rolling(12, min_periods=4).mean()))
    df["value_trend_6_24"] = df["lag_6"] - df["target_value"]
    df["value_volatility_6"] = (df["rolling_max_6"] - df["rolling_min_6"]).abs()
    df["value_volatility_12"] = (
        df.groupby(group_keys)["target_value"]
        .transform(lambda s: s.rolling(12, min_periods=3).std())
    )
    df["value_volatility_24"] = (
        df.groupby(group_keys)["target_value"]
        .transform(lambda s: s.rolling(24, min_periods=6).std())
    )
    df["value_vol_ratio_6_24"] = df["value_volatility_6"] / df["value_volatility_24"].replace(0, np.nan)
    df["value_range_6"] = df["rolling_max_6"] - df["rolling_min_6"]
    df["value_range_12"] = (
        df.groupby(group_keys)["target_value"]
        .transform(lambda s: s.rolling(12, min_periods=3).max() - s.rolling(12, min_periods=3).min())
    )
    df["value_accel_1_3"] = df["lag_1"] - df["lag_3"]
    df["value_accel_3_6"] = df["lag_3"] - df["lag_6"]

    df["weather_temp_trend_3_12"] = df["temp_lag_6"] - df["temp_lag_12"]
    df["weather_precip_intensity_6_24"] = df["precip_roll_sum_12"] / df["precip_roll_sum_24"].replace(0, np.nan)
    df["weather_wind_trend_3_12"] = df["wind_lag_6"] - df["wind_roll_mean_12"]
    df["weather_pressure_trend_3_12"] = df["pressure_delta_3"] - df["pressure_lag_3"]

    return df


def prepare_dataframe() -> pd.DataFrame:
    df = load_bigquery_table(TABLE_NAME)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df[TARGET_COLUMN].notna()].copy()
    df = df.sort_values(["station_name", "timeseries_name", "timestamp_utc"]).reset_index(drop=True)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    for col in BASE_NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_regime_features(df)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in REGIME_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)
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


def main():
    df = prepare_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    fold_predictions = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        meta_test = df.iloc[test_idx][["station_name", "timestamp_utc"]].copy()

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = regression_metrics(y_test, y_pred)
        metrics["fold"] = fold_num
        metrics["train_rows"] = int(len(train_idx))
        metrics["test_rows"] = int(len(test_idx))
        metrics["train_start"] = str(df.iloc[train_idx]["timestamp_utc"].min())
        metrics["train_end"] = str(df.iloc[train_idx]["timestamp_utc"].max())
        metrics["test_start"] = str(df.iloc[test_idx]["timestamp_utc"].min())
        metrics["test_end"] = str(df.iloc[test_idx]["timestamp_utc"].max())
        fold_metrics.append(metrics)

        fold_df = meta_test.copy()
        fold_df[TARGET_COLUMN] = y_test.values
        fold_df["prediction"] = y_pred
        fold_df["fold"] = fold_num
        fold_predictions.append(add_error_columns(fold_df))

    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    station_metrics_df = station_level_metrics(predictions_df)

    full_pipeline = build_pipeline()
    full_pipeline.fit(X, y)

    sample_n = min(1000, len(df))
    perm = permutation_importance(
        full_pipeline,
        X.tail(sample_n),
        y.tail(sample_n),
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    summary = {
        "mean_mae": float(fold_metrics_df["mae"].mean()),
        "mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "std_rmse": float(fold_metrics_df["rmse"].std(ddof=0)),
        "mean_r2": float(fold_metrics_df["r2"].mean()),
        "max_fold_rmse": float(fold_metrics_df["rmse"].max()),
        "rows_scored": int(len(predictions_df)),
    }

    joblib.dump(full_pipeline, OUTPUT_DIR / "gauge_24h_regime_model.joblib")

    with open(OUTPUT_DIR / "gauge_24h_regime_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fold_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_regime_fold_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "gauge_24h_regime_predictions.csv", index=False)
    station_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_regime_station_metrics.csv", index=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "gauge_24h_regime_feature_importance.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()