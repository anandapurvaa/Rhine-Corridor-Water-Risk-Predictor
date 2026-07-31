from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.data_loader import load_bigquery_table


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "dataset_splits_gauge_24h"
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
    categorical_features = CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS
    numeric_features = NUMERIC_COLUMNS

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
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
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
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = load_bigquery_table(TABLE_NAME)

    df["split_name"] = df["split_name"].astype(str).str.strip().str.lower()
    df = df[df[TARGET_COLUMN].notna()].copy()

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "split_name", "timestamp_utc"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in CATEGORICAL_COLUMNS + IDENTIFIER_COLUMNS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = df[df["split_name"] == "train"].copy()
    validation_df = df[df["split_name"].isin(["validation", "val"])].copy()
    test_df = df[df["split_name"] == "test"].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_validation)
    test_pred = pipeline.predict(X_test)

    metrics = {
        "validation": regression_metrics(y_validation, val_pred),
        "test": regression_metrics(y_test, test_pred),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
    }

    predictions_df = test_df[
        ["station_name", "timestamp_utc", TARGET_COLUMN]
    ].copy()
    predictions_df["prediction"] = test_pred
    predictions_df["abs_error"] = (
        predictions_df[TARGET_COLUMN] - predictions_df["prediction"]
    ).abs()

    joblib.dump(pipeline, OUTPUT_DIR / "baseline_gauge_24h_regressor.joblib")

    with open(
        OUTPUT_DIR / "baseline_gauge_24h_regression_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    predictions_df.to_csv(
        OUTPUT_DIR / "baseline_gauge_24h_test_predictions.csv",
        index=False,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()