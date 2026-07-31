from pathlib import Path
import json

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

BASE_CATEGORICAL_COLUMNS = [
    "station_name",
    "timeseries_name",
    "unit",
    "source",
]

BASE_IDENTIFIER_COLUMNS = [
    "station_id",
    "dwd_station_id",
    "dwd_station_name",
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

FEATURE_SETS = {
    "full": {
        "categorical": BASE_CATEGORICAL_COLUMNS + BASE_IDENTIFIER_COLUMNS,
        "numeric": BASE_NUMERIC_COLUMNS,
    },
    "no_target_value": {
        "categorical": BASE_CATEGORICAL_COLUMNS + BASE_IDENTIFIER_COLUMNS,
        "numeric": [c for c in BASE_NUMERIC_COLUMNS if c != "target_value"],
    },
    "no_station_id": {
        "categorical": BASE_CATEGORICAL_COLUMNS,
        "numeric": BASE_NUMERIC_COLUMNS,
    },
    "lean": {
        "categorical": ["station_name", "timeseries_name", "unit", "source"],
        "numeric": [
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
        ],
    },
}


def build_pipeline(categorical_columns, numeric_columns) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_columns,
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

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = load_bigquery_table(TABLE_NAME)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df[TARGET_COLUMN].notna()].copy().sort_values("timestamp_utc").reset_index(drop=True)

    results = []

    for name, cols in FEATURE_SETS.items():
        categorical = cols["categorical"]
        numeric = cols["numeric"]
        feature_columns = categorical + numeric

        missing = [c for c in feature_columns + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

        X = df[feature_columns].copy()
        y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

        for c in categorical:
            X[c] = X[c].astype("string")
        for c in numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        tscv = TimeSeriesSplit(n_splits=5)
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            pipe = build_pipeline(categorical, numeric)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            m = regression_metrics(y.iloc[test_idx], pred)
            m["fold"] = fold
            fold_rows.append(m)

        fold_df = pd.DataFrame(fold_rows)
        results.append(
            {
                "feature_set": name,
                "mean_mae": float(fold_df["mae"].mean()),
                "mean_rmse": float(fold_df["rmse"].mean()),
                "std_rmse": float(fold_df["rmse"].std(ddof=0)),
                "mean_r2": float(fold_df["r2"].mean()),
            }
        )

    results_df = pd.DataFrame(results).sort_values("mean_rmse")
    results_df.to_csv(OUTPUT_DIR / "gauge_24h_feature_ablation.csv", index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()