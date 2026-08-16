from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os

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
    ROBUSTNESS_REQUIRED_COLUMNS,
    TRAIN_TABLE_NAME,
    TARGET_COLUMN as DEFAULT_TARGET_COLUMN,
)

# Shared hyperparameters / config
MIN_REQUIRED_ROWS = 1000
MIN_REQUIRED_STATIONS = 5
HORIZON_HOURS = int(os.getenv("GAUGE24H_HORIZON_HOURS", "24"))
TARGET_MODE = os.getenv("GAUGE24H_TARGET_MODE", "delta").strip().lower()
MAX_FEATURE_NULL_FRACTION = float(os.getenv("GAUGE24H_MAX_FEATURE_NULL_FRACTION", "0.35"))
TRAIN_SPLIT_NAME = os.getenv("GAUGE24H_TRAIN_SPLIT_NAME", "train").strip().lower()


def infer_step_hours(df: pd.DataFrame) -> float:
    diffs = (
        df.sort_values(["station_name", "timestamp_utc"])
        .groupby("station_name")["timestamp_utc"]
        .diff()
        .dropna()
    )
    if diffs.empty:
        raise ValueError("Cannot infer sampling frequency: no timestamp diffs found.")
    return diffs.median().total_seconds() / 3600.0


def build_horizon_target_column(df: pd.DataFrame, horizon_hours: int) -> tuple[pd.DataFrame, str]:
    native_col = f"target_value_t_plus_{horizon_hours}h"
    if native_col in df.columns:
        return df, native_col

    step_hours = infer_step_hours(df)
    if step_hours <= 0:
        raise ValueError(f"Invalid inferred step size: {step_hours} hours")

    shift_steps = round(horizon_hours / step_hours)
    if shift_steps <= 0:
        raise ValueError(f"Horizon {horizon_hours}h is smaller than sampling step {step_hours}h")

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
    seed = (
        f"{TRAIN_TABLE_NAME}|{target_column}|{TARGET_MODE}|{HORIZON_HOURS}|"
        f"{train_start.isoformat()}|{train_end.isoformat()}|{rows}|{TRAIN_SPLIT_NAME}"
    )
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gauge24h_prod__{stamp}__{suffix}"


def resolve_available_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in PRODUCTION_FEATURE_COLUMNS if c in df.columns]


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    numeric_candidates = list(dict.fromkeys(NUMERIC_COLUMNS_LEAN + [DEFAULT_TARGET_COLUMN, "target_value"]))
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_sparse_rows(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    usable_features = [c for c in feature_columns if c in df.columns]
    if not usable_features:
        raise ValueError("No usable feature columns found after schema resolution.")

    null_fraction = df[usable_features].isna().mean(axis=1)
    return df.loc[null_fraction <= MAX_FEATURE_NULL_FRACTION].copy()


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric_features = [c for c in feature_columns if c not in categorical_features]

    transformers = []
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def load_training_frame() -> tuple[pd.DataFrame, str, list[str]]:
    requested = list(
        dict.fromkeys(
            PRODUCTION_FEATURE_COLUMNS
            + [
                DEFAULT_TARGET_COLUMN,
                "target_value",
                "timestamp_utc",
                "split_name",
            ]
        )
    )

    df = load_bigquery_table(
        TRAIN_TABLE_NAME,
        columns=requested,
        where_sql=f"split_name = '{TRAIN_SPLIT_NAME}'",
        order_by="timestamp_utc, station_name",
        allow_missing_columns=True,
    )

    missing_required = [
        c for c in ROBUSTNESS_REQUIRED_COLUMNS
        if c not in df.columns
    ]
    if missing_required:
        raise ValueError(
            f"Missing required base columns: {missing_required}"
        )

    if "split_name" not in df.columns:
        raise ValueError(
            "Training table must include split_name column."
        )

    df["split_name"] = (
        df["split_name"]
        .astype("string")
        .str.lower()
    )

    # Defensive check; BigQuery already filtered this split.
    df = df[df["split_name"] == TRAIN_SPLIT_NAME].copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp_utc"]).copy()

    df = cast_columns(df)
    df = df.dropna(subset=["target_value"]).copy()

    df = df.sort_values(
        ["timestamp_utc", "station_name"]
    ).reset_index(drop=True)

    df, target_column = build_horizon_target_column(
        df,
        HORIZON_HOURS,
    )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )
    df = df.dropna(subset=[target_column]).copy()

    feature_columns = resolve_available_feature_columns(df)
    df = filter_sparse_rows(df, feature_columns)

    df = df.sort_values(
        ["timestamp_utc", "station_name"]
    ).reset_index(drop=True)

    return df, target_column, feature_columns


def validate_training_frame(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> None:
    missing = [c for c in ["timestamp_utc", "target_value", target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after preparation: {missing}")
    if not feature_columns:
        raise ValueError("No feature columns available for training.")
    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Training frame too small: {len(df)} rows < {MIN_REQUIRED_ROWS}")
    if int(df["station_name"].nunique()) < MIN_REQUIRED_STATIONS:
        raise ValueError(f"Too few stations: {int(df['station_name'].nunique())} < {MIN_REQUIRED_STATIONS}")
    if df[target_column].notna().sum() == 0:
        raise ValueError("No non-null target rows available")


def training_null_profile(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> dict:
    cols = list(dict.fromkeys(feature_columns + [target_column, "target_value"]))
    return {col: {"null_rate": float(df[col].isna().mean())} for col in cols if col in df.columns}