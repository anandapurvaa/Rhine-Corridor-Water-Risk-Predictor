from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

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
    TARGET_COLUMN,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"

MIN_REQUIRED_ROWS = 1000
MIN_REQUIRED_STATIONS = 5


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
                NUMERIC_COLUMNS_LEAN,
            ),
        ],
        remainder="drop",
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


def stable_model_version(train_start: pd.Timestamp, train_end: pd.Timestamp, rows: int) -> str:
    seed = f"{TABLE_NAME}|{TARGET_COLUMN}|{train_start.isoformat()}|{train_end.isoformat()}|{rows}"
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gauge24h_prod__{stamp}__{suffix}"


def load_training_frame() -> pd.DataFrame:
    columns = list(dict.fromkeys(PRODUCTION_FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"]))
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
        df[col] = df[col].astype("string")

    for col in NUMERIC_COLUMNS_LEAN + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df.sort_values(["timestamp_utc", "station_name"]).reset_index(drop=True)
    return df


def validate_training_frame(df: pd.DataFrame) -> None:
    missing = [c for c in PRODUCTION_FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp_utc"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Training frame too small: {len(df)} rows < {MIN_REQUIRED_ROWS}")

    station_count = int(df["station_name"].nunique())
    if station_count < MIN_REQUIRED_STATIONS:
        raise ValueError(f"Too few stations: {station_count} < {MIN_REQUIRED_STATIONS}")

    if df[TARGET_COLUMN].notna().sum() == 0:
        raise ValueError("No non-null target rows available")

    if df["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc contains nulls after parsing")


def training_null_profile(df: pd.DataFrame) -> dict:
    cols = PRODUCTION_FEATURE_COLUMNS + [TARGET_COLUMN]
    profile = {}
    for col in cols:
        profile[col] = {
            "null_rate": float(df[col].isna().mean()),
        }
    return profile


def main():
    df = load_training_frame()
    validate_training_frame(df)

    X = df[PRODUCTION_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    model = build_pipeline()
    model.fit(X, y)

    train_start = df["timestamp_utc"].min()
    train_end = df["timestamp_utc"].max()
    model_version = stable_model_version(train_start, train_end, len(df))

    joblib.dump(model, MODEL_PATH)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "table_name": TABLE_NAME,
        "target_column": TARGET_COLUMN,
        "feature_columns": PRODUCTION_FEATURE_COLUMNS,
        "rows_trained": int(len(df)),
        "stations_trained": int(df["station_name"].nunique()),
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "model_path": str(MODEL_PATH),
        "null_profile": training_null_profile(df),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Production training complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()