from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

IDENTIFIER_COLUMNS = [
    "station_id",
    "dwd_station_id",
    "dwd_station_name",
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
    "diff_1",
    "diff_3",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_min_6",
    "rolling_max_6",
    "hour_utc",
    "day_of_week",
    "month",
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_6",
    "temp_lag_12",
    "precip_lag_1",
    "precip_lag_3",
    "precip_lag_6",
    "precip_lag_12",
    "wind_lag_1",
    "wind_lag_3",
    "wind_lag_6",
    "pressure_lag_1",
    "pressure_lag_3",
    "humidity_lag_1",
    "temp_roll_mean_3",
    "temp_roll_mean_6",
    "temp_roll_mean_12",
    "precip_roll_mean_6",
    "precip_roll_mean_12",
    "precip_roll_sum_6",
    "precip_roll_sum_12",
    "precip_roll_sum_24",
    "wind_roll_mean_6",
    "wind_roll_mean_12",
    "pressure_roll_mean_6",
    "pressure_roll_mean_12",
    "humidity_roll_mean_6",
    "humidity_roll_mean_12",
    "pressure_delta_1",
    "pressure_delta_3",
    "temp_change_1_3",
    "precip_accel_12_24",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS + NUMERIC_COLUMNS


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
                CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS,
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
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
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
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def station_level_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_name, g in scored_df.groupby("station_name", dropna=False):
        y_true = g[TARGET_COLUMN]
        y_pred = g["prediction"]
        metrics = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "station_name": station_name,
                "rows": int(len(g)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def fold_level_station_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, station_name), g in scored_df.groupby(["fold", "station_name"], dropna=False):
        y_true = g[TARGET_COLUMN]
        y_pred = g["prediction"]
        metrics = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "fold": int(fold),
                "station_name": station_name,
                "rows": int(len(g)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "rmse"], ascending=[True, False]).reset_index(drop=True)


def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["residual"] = pred_df[TARGET_COLUMN] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["squared_error"] = pred_df["residual"] ** 2
    return pred_df


def save_diagnostic_plots(predictions_df: pd.DataFrame, fold_metrics_df: pd.DataFrame) -> None:
    plot_df = predictions_df.dropna(subset=[TARGET_COLUMN, "prediction", "residual"]).copy()

    sample_n = min(2500, len(plot_df))
    plot_sample = plot_df.sample(sample_n, random_state=42) if len(plot_df) > sample_n else plot_df

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_sample[TARGET_COLUMN], plot_sample["prediction"], alpha=0.35, s=16)
    mn = min(plot_sample[TARGET_COLUMN].min(), plot_sample["prediction"].min())
    mx = max(plot_sample[TARGET_COLUMN].max(), plot_sample["prediction"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_sample["prediction"], plot_sample["residual"], alpha=0.35, s=16)
    plt.axhline(y=0.0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_residuals_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(plot_df["residual"], bins=50)
    plt.axvline(x=0.0, linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_residual_distribution.png", dpi=180)
    plt.close()

    time_df = plot_df.sort_values("timestamp_utc").copy()
    rolling = (
        time_df.set_index("timestamp_utc")["abs_error"]
        .rolling("14D", min_periods=20)
        .mean()
        .dropna()
    )
    plt.figure(figsize=(10, 5))
    plt.plot(rolling.index, rolling.values)
    plt.xlabel("Timestamp")
    plt.ylabel("14-day Rolling Mean Absolute Error")
    plt.title("Residual Drift Over Time")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_residual_drift_over_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(fold_metrics_df["fold"].astype(str), fold_metrics_df["rmse"])
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title("Fold RMSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_fold_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(fold_metrics_df["fold"].astype(str), fold_metrics_df["mae"])
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Fold MAE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauge_24h_fold_mae.png", dpi=180)
    plt.close()


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
    fold_station_metrics_df = fold_level_station_metrics(predictions_df)

    full_pipeline = build_pipeline()
    full_pipeline.fit(X, y)

    sample_n = min(1000, len(df))
    X_sample = X.tail(sample_n)
    y_sample = y.tail(sample_n)

    perm = permutation_importance(
        full_pipeline,
        X_sample,
        y_sample,
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
        "backtest_mean_mae": float(fold_metrics_df["mae"].mean()),
        "backtest_mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "backtest_mean_r2": float(fold_metrics_df["r2"].mean()),
        "backtest_std_rmse": float(fold_metrics_df["rmse"].std(ddof=0)),
        "backtest_max_fold_rmse": float(fold_metrics_df["rmse"].max()),
        "backtest_min_fold_rmse": float(fold_metrics_df["rmse"].min()),
        "mean_residual": float(predictions_df["residual"].mean()),
        "p90_abs_error": float(predictions_df["abs_error"].quantile(0.90)),
        "max_abs_error": float(predictions_df["abs_error"].max()),
        "folds": int(len(fold_metrics_df)),
        "rows_scored": int(len(predictions_df)),
    }

    save_diagnostic_plots(predictions_df, fold_metrics_df)

    joblib.dump(full_pipeline, OUTPUT_DIR / "gauge_24h_backtest_full_model.joblib")

    with open(OUTPUT_DIR / "gauge_24h_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fold_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_backtest_fold_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "gauge_24h_backtest_predictions.csv", index=False)
    station_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_backtest_station_metrics.csv", index=False)
    fold_station_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_backtest_fold_station_metrics.csv", index=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "gauge_24h_backtest_feature_importance.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()