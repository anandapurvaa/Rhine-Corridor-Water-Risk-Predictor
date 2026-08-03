from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.data_loader import load_bigquery_table
from modeling.schemas import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS_LEAN,
    PRODUCTION_FEATURE_COLUMNS,
    TABLE_NAME,
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"

MIN_REQUIRED_ROWS = 1000
MIN_REQUIRED_STATIONS = 5
HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
TARGET_MODE = os.getenv("GAUGE24H_TARGET_MODE", "level").strip().lower()


def build_pipeline() -> Pipeline:
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in PRODUCTION_FEATURE_COLUMNS]
    numeric_features = [c for c in PRODUCTION_FEATURE_COLUMNS if c not in categorical_features]

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
        sparse_threshold=0,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
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


def build_training_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    y_level = pd.to_numeric(df[target_column], errors="coerce")
    if TARGET_MODE == "delta":
        y_now = pd.to_numeric(df["target_value"], errors="coerce")
        return y_level - y_now
    if TARGET_MODE == "level":
        return y_level
    raise ValueError(f"Unsupported GAUGE24H_TARGET_MODE={TARGET_MODE!r}")


def stable_model_version(train_start: pd.Timestamp, train_end: pd.Timestamp, rows: int, target_column: str) -> str:
    seed = f"{TABLE_NAME}|{target_column}|{TARGET_MODE}|{HORIZON_HOURS}|{train_start.isoformat()}|{train_end.isoformat()}|{rows}"
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gauge24h_prod__{stamp}__{suffix}"


def load_training_frame() -> tuple[pd.DataFrame, str]:
    columns = list(dict.fromkeys(PRODUCTION_FEATURE_COLUMNS + [DEFAULT_TARGET_COLUMN, "target_value", "timestamp_utc"]))
    df = load_bigquery_table(
        TABLE_NAME,
        columns=columns,
        order_by="timestamp_utc, station_name",
    )

    if "timestamp_utc" not in df.columns:
        raise ValueError("Expected timestamp_utc in training table")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in list(dict.fromkeys(NUMERIC_COLUMNS_LEAN + [DEFAULT_TARGET_COLUMN, "target_value"])):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_value"]).copy()
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)

    df, target_column = build_horizon_target_column(df, HORIZON_HOURS)
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column]).copy()
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)

    return df, target_column


def validate_training_frame(df: pd.DataFrame, target_column: str) -> None:
    missing = [c for c in PRODUCTION_FEATURE_COLUMNS + [target_column, "timestamp_utc", "target_value"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Training frame too small: {len(df)} rows < {MIN_REQUIRED_ROWS}")

    station_count = int(df["station_name"].nunique())
    if station_count < MIN_REQUIRED_STATIONS:
        raise ValueError(f"Too few stations: {station_count} < {MIN_REQUIRED_STATIONS}")

    if df[target_column].notna().sum() == 0:
        raise ValueError("No non-null target rows available")

    if df["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc contains nulls after parsing")


def training_null_profile(df: pd.DataFrame, target_column: str) -> dict:
    cols = list(dict.fromkeys(PRODUCTION_FEATURE_COLUMNS + [target_column, "target_value"]))
    profile = {}
    for col in cols:
        if col in df.columns:
            profile[col] = {"null_rate": float(df[col].isna().mean())}
    return profile


def main():
    df, target_column = load_training_frame()
    validate_training_frame(df, target_column)

    X = df[PRODUCTION_FEATURE_COLUMNS].copy()
    y = build_training_target(df, target_column)

    model = build_pipeline()
    model.fit(X, y)

    train_start = df["timestamp_utc"].min()
    train_end = df["timestamp_utc"].max()
    model_version = stable_model_version(train_start, train_end, len(df), target_column)

    joblib.dump(model, MODEL_PATH)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "table_name": TABLE_NAME,
        "target_column": target_column,
        "target_mode": TARGET_MODE,
        "horizon_hours": HORIZON_HOURS,
        "feature_columns": PRODUCTION_FEATURE_COLUMNS,
        "rows_trained": int(len(df)),
        "stations_trained": int(df["station_name"].nunique()),
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "model_path": str(MODEL_PATH),
        "null_profile": training_null_profile(df, target_column),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Production training complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()