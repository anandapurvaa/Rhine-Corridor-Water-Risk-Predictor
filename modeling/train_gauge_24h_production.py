from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

from modeling.schemas import (
    PRODUCTION_FEATURE_COLUMNS,
    TRAIN_TABLE_NAME,
)
from modeling.training_utils import (
    HORIZON_HOURS,
    TARGET_MODE,
    MAX_FEATURE_NULL_FRACTION,
    TRAIN_SPLIT_NAME,
    build_training_target,
    stable_model_version,
    load_training_frame,
    validate_training_frame,
    training_null_profile,
    build_pipeline,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gauge_24h_production_model.joblib"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "gauge_24h_production_training_summary.json"


def main():
    df, target_column, feature_columns = load_training_frame()
    validate_training_frame(df, target_column, feature_columns)

    X = df[feature_columns].copy()
    y = build_training_target(df, target_column)

    model = build_pipeline(feature_columns)
    model.fit(X, y)

    train_start = df["timestamp_utc"].min()
    train_end = df["timestamp_utc"].max()
    model_version = stable_model_version(train_start, train_end, len(df), target_column)

    joblib.dump(model, MODEL_PATH)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "table_name": TRAIN_TABLE_NAME,
        "train_split_used": TRAIN_SPLIT_NAME,
        "target_column": target_column,
        "target_mode": TARGET_MODE,
        "horizon_hours": HORIZON_HOURS,
        "feature_columns_requested": PRODUCTION_FEATURE_COLUMNS,
        "feature_columns_used": feature_columns,
        "rows_trained": int(len(df)),
        "stations_trained": int(df["station_name"].nunique()),
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "model_path": str(MODEL_PATH),
        "max_feature_null_fraction": MAX_FEATURE_NULL_FRACTION,
        "null_profile": training_null_profile(df, feature_columns, target_column),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Production training complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()