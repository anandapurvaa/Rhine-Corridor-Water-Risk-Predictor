from pathlib import Path
import json

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


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true, y_pred):
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


def prepare_dataframe():
    df = load_bigquery_table(TABLE_NAME)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df[TARGET_COLUMN].notna()].copy().sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype("string")
    for c in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    df = prepare_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    fold_predictions = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        meta = df.iloc[test_idx][["station_name", "timestamp_utc"]].copy()

        pipe = build_pipeline()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        m = regression_metrics(y_test, pred)
        m["fold"] = fold_num
        m["train_rows"] = int(len(train_idx))
        m["test_rows"] = int(len(test_idx))
        m["train_start"] = str(df.iloc[train_idx]["timestamp_utc"].min())
        m["train_end"] = str(df.iloc[train_idx]["timestamp_utc"].max())
        m["test_start"] = str(df.iloc[test_idx]["timestamp_utc"].min())
        m["test_end"] = str(df.iloc[test_idx]["timestamp_utc"].max())
        fold_metrics.append(m)

        pred_df = meta.copy()
        pred_df[TARGET_COLUMN] = y_test.values
        pred_df["prediction"] = pred
        pred_df["residual"] = pred_df[TARGET_COLUMN] - pred_df["prediction"]
        pred_df["abs_error"] = pred_df["residual"].abs()
        pred_df["fold"] = fold_num
        fold_predictions.append(pred_df)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(fold_predictions, ignore_index=True)

    full_pipe = build_pipeline()
    full_pipe.fit(X, y)

    perm = permutation_importance(
        full_pipe,
        X.tail(min(1000, len(X))),
        y.tail(min(1000, len(y))),
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    feat_imp = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    fold_metrics_df.to_csv(OUTPUT_DIR / "gauge_24h_lean_fold_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "gauge_24h_lean_predictions.csv", index=False)
    feat_imp.to_csv(OUTPUT_DIR / "gauge_24h_lean_feature_importance.csv", index=False)

    summary = {
        "mean_mae": float(fold_metrics_df["mae"].mean()),
        "mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "std_rmse": float(fold_metrics_df["rmse"].std(ddof=0)),
        "mean_r2": float(fold_metrics_df["r2"].mean()),
        "rows_scored": int(len(pred_df)),
    }

    with open(OUTPUT_DIR / "gauge_24h_lean_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()