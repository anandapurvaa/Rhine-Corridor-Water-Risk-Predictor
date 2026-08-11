from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

import numpy as np
import pandas as pd

from modeling.data_loader import (
    load_bigquery_table,
    write_bigquery_table,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURATED_DATASET = os.getenv(
    "CURATED_DATASET",
    "rhein_curated",
).strip()

EVALUATION_KEY_COLUMNS = [
    "split_name",
    "run_id",
    "station_name",
    "forecast_timestamp_utc",
    "model_version",
]

EVALUATION_TABLE = os.getenv(
    "EVALUATIONS_TABLE",
    "gauge_24h_prediction_evaluations",
).strip()

EVALUATION_CSV = (
    OUTPUT_DIR
    / "gauge_24h_prediction_evaluation.csv"
)

EVALUATION_SUMMARY_JSON = (
    OUTPUT_DIR
    / "gauge_24h_prediction_evaluation_summary.json"
)

STATION_METRICS_CSV = (
    OUTPUT_DIR
    / "gauge_24h_prediction_evaluation_station_metrics.csv"
)

MODEL_METRICS_CSV = (
    OUTPUT_DIR
    / "gauge_24h_prediction_evaluation_model_metrics.csv"
)

PREDICTION_HISTORY_TABLE = os.getenv(
    "PREDICTIONS_HISTORY_TABLE",
    "gauge_24h_prediction_history",
).strip()

PREDICTIONS_TEST_TABLE = os.getenv(
    "PREDICTIONS_TEST_TABLE",
    "gauge_24h_production_predictions_test",
).strip()

PREDICTIONS_VALIDATION_TABLE = os.getenv(
    "PREDICTIONS_VALIDATION_TABLE",
    "gauge_24h_production_predictions_validation",
).strip()

ACTUALS_TABLE = os.getenv(
    "GAUGE24H_ACTUALS_TABLE",
    "pegelonline_measurements_curated",
).strip()

DATASET_SPLITS_TABLE = os.getenv(
    "GAUGE24H_DATASET_SPLITS_TABLE",
    "dataset_splits_gauge_24h",
).strip()

THRESHOLD_CONFIG_PATH = Path(
    os.getenv(
        "GAUGE24H_THRESHOLD_CONFIG_PATH",
        "config/thresholds.yaml",
    ).strip()
)

ACTUAL_MATCH_TOLERANCE_MINUTES = int(
    os.getenv(
        "GAUGE24H_ACTUAL_MATCH_TOLERANCE_MINUTES",
        "30",
    )
)

EVAL_SPLIT_NAME = os.getenv(
    "GAUGE24H_EVAL_SPLIT_NAME",
    "test",
).strip().lower()

if EVAL_SPLIT_NAME not in {
    "production",
    "validation",
    "test",
}:
    raise ValueError(
        "GAUGE24H_EVAL_SPLIT_NAME must be "
        "'production', 'validation', or 'test'"
    )

if EVAL_SPLIT_NAME == "production":
    PREDICTIONS_TABLE = PREDICTION_HISTORY_TABLE
elif EVAL_SPLIT_NAME == "test":
    PREDICTIONS_TABLE = PREDICTIONS_TEST_TABLE
else:
    PREDICTIONS_TABLE = PREDICTIONS_VALIDATION_TABLE

def normalize_utc_timestamp_dtype(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

def load_thresholds() -> dict[str, float]:
    if not THRESHOLD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Threshold config not found: {THRESHOLD_CONFIG_PATH}"
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config/threshold.yaml"
        ) from exc

    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    thresholds = data.get("low_water_thresholds_cm", data)
    if not isinstance(thresholds, dict):
        raise ValueError(
            "Threshold config must contain a mapping under "
            "'low_water_thresholds_cm'."
        )

    normalized: dict[str, float] = {}
    for station, value in thresholds.items():
        if value is None:
            continue
        normalized[str(station).strip().upper()] = float(value)

    if not normalized:
        raise ValueError("No thresholds found in threshold config.")

    return normalized


THRESHOLD_BY_STATION = load_thresholds()


def normalize_station_name(name: str) -> str:
    return str(name).strip().upper()


def load_predictions() -> pd.DataFrame:
    df = load_bigquery_table(
        PREDICTIONS_TABLE,
        dataset=CURATED_DATASET,
        order_by=(
            "split_name, "
            "forecast_timestamp_utc, "
            "station_name"
        ),
    )

    if df.empty:
        raise ValueError(
            "No rows found in "
            f"{CURATED_DATASET}.{PREDICTIONS_TABLE}"
        )

    required = [
        "station_name",
        "prediction",
        "split_name",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Predictions table missing required "
            f"columns: {missing}"
        )

    for column in [
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction_ready_utc",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce",
            )

    if "prediction_horizon_hours" in df.columns:
        df["prediction_horizon_hours"] = pd.to_numeric(
            df["prediction_horizon_hours"],
            errors="coerce",
        )

    if "timestamp_utc" in df.columns:
        if "forecast_timestamp_utc" not in df.columns:
            horizon_hours = 24

            horizon_values = (
                df["prediction_horizon_hours"]
                .dropna()
                .unique()
                .tolist()
                if "prediction_horizon_hours" in df.columns
                else []
            )

            if horizon_values:
                horizon_hours = int(horizon_values[0])

            df["forecast_timestamp_utc"] = (
                df["timestamp_utc"]
                + pd.to_timedelta(
                    horizon_hours,
                    unit="h",
                )
            )
    else:
        raise ValueError(
            "Predictions table must include "
            "'timestamp_utc' or "
            "'forecast_timestamp_utc'."
        )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    df["split_name"] = (
        df["split_name"]
        .astype("string")
        .str.lower()
    )

    df["station_name"] = (
        df["station_name"]
        .astype("string")
    )

    df = df.dropna(
        subset=[
            "station_name",
            "prediction",
            "forecast_timestamp_utc",
        ]
    ).copy()

    return df


def load_split_actuals() -> pd.DataFrame:
    df = load_bigquery_table(
        DATASET_SPLITS_TABLE,
        dataset=CURATED_DATASET,
        order_by=(
            "split_name, "
            "timestamp_utc, "
            "station_name"
        ),
    )

    if df.empty:
        raise ValueError(
            "No rows found in "
            f"{CURATED_DATASET}.{DATASET_SPLITS_TABLE}"
        )

    required = [
        "split_name",
        "station_name",
        "timestamp_utc",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset splits table missing required "
            f"columns: {missing}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    df["station_name"] = (
        df["station_name"]
        .astype("string")
    )

    df["split_name"] = (
        df["split_name"]
        .astype("string")
        .str.lower()
    )

    numeric_columns = [
        column
        for column in df.columns
        if column.startswith(
            "target_value_t_plus_"
        )
    ]

    numeric_columns.append("target_value")

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def load_production_actuals() -> pd.DataFrame:
    df = load_bigquery_table(
        ACTUALS_TABLE,
        dataset=CURATED_DATASET,
        columns=[
            "station_name",
            "timestamp_utc",
            "value",
            "unit",
            "source",
        ],
        order_by=(
            "timestamp_utc, "
            "station_name"
        ),
        allow_missing_columns=True,
    )

    if df.empty:
        raise ValueError(
            "No rows found in "
            f"{CURATED_DATASET}.{ACTUALS_TABLE}"
        )

    required = [
        "station_name",
        "timestamp_utc",
        "value",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Production actuals table missing "
            f"required columns: {missing}"
        )

    df["station_name"] = (
        df["station_name"]
        .astype("string")
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "station_name",
            "timestamp_utc",
            "value",
        ]
    ).copy()

    df = df.sort_values(
        [
            "station_name",
            "timestamp_utc",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "station_name",
            "timestamp_utc",
        ],
        keep="last",
    )

    return df[
        [
            "station_name",
            "timestamp_utc",
            "value",
        ]
    ].reset_index(drop=True)


def resolve_actual_values(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    if target_column not in actual_df.columns:
        raise ValueError(
            f"Target column {target_column!r} "
            "not found in actuals table."
        )

    actual_long = actual_df[
        [
            "split_name",
            "station_name",
            "timestamp_utc",
            target_column,
        ]
    ].copy()

    actual_long = actual_long.rename(
        columns={
            "timestamp_utc": (
                "actual_timestamp_utc"
            ),
            target_column: "actual_value",
        }
    )

    actual_long["target_column"] = (
        target_column
    )

    merged = pred_df.merge(
        actual_long,
        left_on=[
            "split_name",
            "station_name",
            "forecast_timestamp_utc",
            "target_column",
        ],
        right_on=[
            "split_name",
            "station_name",
            "actual_timestamp_utc",
            "target_column",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged


def resolve_production_actual_values(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pred_df.copy()
    actuals = actual_df.copy()

    predictions["forecast_timestamp_utc"] = (
        pd.to_datetime(
            predictions["forecast_timestamp_utc"],
            utc=True,
            errors="coerce",
        ).astype("datetime64[ns, UTC]")
    )

    actuals["timestamp_utc"] = (
        pd.to_datetime(
            actuals["timestamp_utc"],
            utc=True,
            errors="coerce",
        ).astype("datetime64[ns, UTC]")
    )

    actuals = actuals.rename(
        columns={
            "value": "actual_value",
            "timestamp_utc": "actual_timestamp_utc",
        }
    )

    predictions = predictions.dropna(
        subset=[
            "station_name",
            "forecast_timestamp_utc",
        ]
    ).copy()

    actuals = actuals.dropna(
        subset=[
            "station_name",
            "actual_timestamp_utc",
            "actual_value",
        ]
    ).copy()

    predictions = predictions.sort_values(
        [
            "station_name",
            "forecast_timestamp_utc",
        ]
    )

    actuals = actuals.sort_values(
        [
            "station_name",
            "actual_timestamp_utc",
        ]
    )

    matched_groups = []

    for station_name, station_predictions in (
        predictions.groupby(
            "station_name",
            dropna=False,
        )
    ):
        station_actuals = actuals[
            actuals["station_name"] == station_name
        ].copy()

        if station_actuals.empty:
            group = station_predictions.copy()
            group["actual_timestamp_utc"] = pd.NaT
            group["actual_value"] = np.nan
            matched_groups.append(group)
            continue

        station_predictions = (
            station_predictions
            .sort_values("forecast_timestamp_utc")
            .reset_index(drop=True)
        )

        station_actuals = (
            station_actuals
            .sort_values("actual_timestamp_utc")
            .reset_index(drop=True)
        )

        group = pd.merge_asof(
            station_predictions,
            station_actuals,
            left_on="forecast_timestamp_utc",
            right_on="actual_timestamp_utc",
            direction="nearest",
            tolerance=pd.Timedelta(
                minutes=ACTUAL_MATCH_TOLERANCE_MINUTES
            ),
            suffixes=(
                "",
                "_actual_source",
            ),
        )

        matched_groups.append(group)

    if not matched_groups:
        result = predictions.copy()
        result["actual_timestamp_utc"] = pd.NaT
        result["actual_value"] = np.nan
    else:
        result = pd.concat(
            matched_groups,
            ignore_index=True,
        )

    result["actual_available"] = (
        result["actual_value"].notna()
    )

    result["actual_match_tolerance_minutes"] = (
        ACTUAL_MATCH_TOLERANCE_MINUTES
    )

    return result


def format_timedelta_human(
    delta: pd.Timedelta | None,
) -> str | None:
    if delta is None or pd.isna(delta):
        return None

    total_seconds = int(
        max(delta.total_seconds(), 0)
    )

    days, remainder = divmod(
        total_seconds,
        86400,
    )

    hours, remainder = divmod(
        remainder,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    parts = []

    if days:
        parts.append(f"{days}d")

    if hours or days:
        parts.append(f"{hours}h")

    if minutes or hours or days:
        parts.append(f"{minutes}m")

    parts.append(f"{seconds}s")

    return " ".join(parts)


def build_not_ready_summary(
    pred_df: pd.DataFrame,
    reason: str,
    split_name: str,
    target_column: str | None = None,
) -> dict:
    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    forecast_min = (
        pred_df["forecast_timestamp_utc"].min()
        if len(pred_df)
        else None
    )

    forecast_max = (
        pred_df["forecast_timestamp_utc"].max()
        if len(pred_df)
        else None
    )

    horizon_hours = None

    if "prediction_horizon_hours" in pred_df.columns:
        horizon_values = pd.to_numeric(
            pred_df[
                "prediction_horizon_hours"
            ],
            errors="coerce",
        ).dropna()

        if not horizon_values.empty:
            horizon_hours = int(
                horizon_values.min()
            )

    evaluation_available_at = None
    time_until_ready = None

    if (
        forecast_max is not None
        and horizon_hours is not None
    ):
        evaluation_available_at = (
            forecast_max
            + pd.Timedelta(
                hours=horizon_hours
            )
        )
        time_until_ready = (
            evaluation_available_at
            - now_utc
        )

    elif forecast_max is not None:
        evaluation_available_at = forecast_max
        time_until_ready = (
            evaluation_available_at
            - now_utc
        )

    human_wait = format_timedelta_human(
        time_until_ready
    )

    wait_seconds = (
        max(
            int(
                time_until_ready.total_seconds()
            ),
            0,
        )
        if time_until_ready is not None
        else None
    )

    message = reason

    if evaluation_available_at is not None:
        message = (
            f"{reason} Evaluation can be "
            f"performed at "
            f"{evaluation_available_at.isoformat()}."
        )

    if (
        human_wait is not None
        and wait_seconds
        and wait_seconds > 0
    ):
        message += (
            f" Time remaining: "
            f"{human_wait}."
        )

    return {
        "evaluated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": "not_ready",
        "split_name": split_name,
        "target_column": target_column,
        "reason": reason,
        "message": message,
        "source_predictions_table": (
            PREDICTIONS_TABLE
        ),
        "source_actuals_table": (
            ACTUALS_TABLE
            if split_name == "production"
            else DATASET_SPLITS_TABLE
        ),
        "rows_in_predictions_table": int(
            len(pred_df)
        ),
        "model_versions_present": (
            int(
                pred_df[
                    "model_version"
                ].nunique()
            )
            if (
                "model_version"
                in pred_df.columns
                and len(pred_df)
            )
            else 0
        ),
        "forecast_min_utc": (
            forecast_min.isoformat()
            if forecast_min is not None
            else None
        ),
        "forecast_max_utc": (
            forecast_max.isoformat()
            if forecast_max is not None
            else None
        ),
        "current_time_utc": (
            now_utc.isoformat()
        ),
        "evaluation_available_at_utc": (
            evaluation_available_at.isoformat()
            if evaluation_available_at is not None
            else None
        ),
        "time_until_evaluation_seconds": (
            wait_seconds
        ),
        "time_until_evaluation_human": (
            human_wait
        ),
        "actual_match_tolerance_minutes": (
            ACTUAL_MATCH_TOLERANCE_MINUTES
            if split_name == "production"
            else None
        ),
    }


def prepare_evaluation_frame(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    pred_df = pred_df[
        pred_df["split_name"]
        == split_name
    ].copy()

    if pred_df.empty:
        raise RuntimeError(
            f"No predictions found for "
            f"split={split_name!r}"
        )

    if split_name == "production":
        mature_pred = pred_df[
            pred_df["forecast_timestamp_utc"]
            <= now_utc
        ].copy()

        if mature_pred.empty:
            raise RuntimeError(
                "No matured production "
                "predictions yet; the forecast "
                "horizon has not elapsed."
            )

        merged = (
            resolve_production_actual_values(
                mature_pred,
                actual_df,
            )
        )

        matched = int(
            merged["actual_value"]
            .notna()
            .sum()
        )

        if matched == 0:
            raise RuntimeError(
                "No production actuals matched "
                "within the configured tolerance "
                f"of {ACTUAL_MATCH_TOLERANCE_MINUTES} "
                "minutes."
            )

        merged["actual_available"] = (
            merged["actual_value"].notna()
        )

        merged = merged[
            merged["actual_available"]
        ].copy()

    else:
        if "prediction_ready_utc" in pred_df.columns:
            latest_run_ts = (
                pred_df[
                    "prediction_ready_utc"
                ].max()
            )

            pred_df = pred_df[
                pred_df[
                    "prediction_ready_utc"
                ]
                == latest_run_ts
            ].copy()

        mature_pred = pred_df[
            pred_df["forecast_timestamp_utc"]
            <= now_utc
        ].copy()

        if mature_pred.empty:
            raise RuntimeError(
                "No matured predictions yet; "
                "the forecast horizon has not "
                "elapsed for this split."
            )

        if "target_column" not in mature_pred.columns:
            raise RuntimeError(
                "Predictions do not have a "
                "target_column field."
            )

        target_column = str(
            mature_pred[
                "target_column"
            ].iloc[0]
        )

        split_actuals = actual_df[
            actual_df["split_name"]
            == split_name
        ].copy()

        merged = resolve_actual_values(
            mature_pred,
            split_actuals,
            target_column,
        )

        matched = int(
            merged["actual_value"]
            .notna()
            .sum()
        )

        if matched == 0:
            raise RuntimeError(
                f"No actuals matched for "
                f"target_column={target_column!r} "
                f"and split={split_name!r}. "
                "Check that dataset_splits_gauge_24h "
                "contains non-null target values."
            )

        merged["actual_available"] = (
            merged["actual_value"].notna()
        )

        merged = merged[
            merged["actual_available"]
        ].copy()

    merged["error"] = (
        merged["actual_value"]
        - merged["prediction"]
    )

    merged["abs_error"] = (
        merged["error"].abs()
    )

    merged["absolute_error"] = (
        merged["abs_error"]
    )

    merged["squared_error"] = (
        merged["error"] ** 2
    )

    merged["ape"] = np.where(
        merged["actual_value"].abs()
        > 1e-9,
        merged["abs_error"]
        / merged["actual_value"].abs(),
        np.nan,
    )

    merged["station_name"] = merged["station_name"].map(normalize_station_name)
    merged["threshold"] = (
        merged["station_name"]
        .map(THRESHOLD_BY_STATION)
    )

    merged["actual_event_low_water"] = np.where(
        merged["threshold"].notna(),
        merged["actual_value"]
        <= merged["threshold"],
        pd.NA,
    )

    merged["pred_event_low_water"] = np.where(
        merged["threshold"].notna(),
        merged["prediction"]
        <= merged["threshold"],
        pd.NA,
    )

    merged["evaluated_at_utc"] = (
        pd.Timestamp.now(tz="UTC")
    )

    return merged.sort_values(
        [
            "forecast_timestamp_utc",
            "station_name",
        ]
    ).reset_index(drop=True)


def regression_metrics(
    df: pd.DataFrame,
) -> dict:
    if df.empty:
        return {
            "rows_evaluated": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape": None,
            "p90_abs_error": None,
        }

    mse = float(
        df["squared_error"].mean()
    )

    rmse = float(np.sqrt(mse))
    mae = float(
        df["abs_error"].mean()
    )
    bias = float(
        df["error"].mean()
    )

    mape = (
        float(
            df["ape"]
            .dropna()
            .mean()
        )
        if df["ape"].notna().any()
        else None
    )

    p90_abs_error = float(
        df["abs_error"].quantile(0.90)
    )

    return {
        "rows_evaluated": int(len(df)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "p90_abs_error": p90_abs_error,
    }


def classification_metrics(
    df: pd.DataFrame,
) -> dict:
    event_df = df.dropna(
        subset=[
            "actual_event_low_water",
            "pred_event_low_water",
        ]
    ).copy()

    if event_df.empty:
        return {
            "event_rows": 0,
            "event_precision": None,
            "event_recall": None,
            "event_f1": None,
        }

    actual_event = (
        event_df[
            "actual_event_low_water"
        ].astype(bool)
    )

    pred_event = (
        event_df[
            "pred_event_low_water"
        ].astype(bool)
    )

    tp = int(
        (actual_event & pred_event).sum()
    )

    fp = int(
        (~actual_event & pred_event).sum()
    )

    fn = int(
        (actual_event & ~pred_event).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "event_rows": int(
            len(event_df)
        ),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def build_station_metrics(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for station_name, group in eval_df.groupby(
        "station_name",
        dropna=False,
    ):
        row = {
            "station_name": station_name
        }

        row.update(
            regression_metrics(group)
        )

        row.update(
            classification_metrics(group)
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "rmse",
        ascending=False,
    ).reset_index(drop=True)


def build_model_metrics(
    eval_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    group_cols = []

    for column in [
        "model_version",
        "run_id",
        "target_mode",
        "prediction_horizon_hours",
        "target_column",
    ]:
        if column in eval_df.columns:
            group_cols.append(column)

    if not group_cols:
        return pd.DataFrame()

    rows = []

    for keys, group in eval_df.groupby(
        group_cols,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            "split_name": split_name
        }

        row.update(
            {
                column: value
                for column, value in zip(
                    group_cols,
                    keys,
                )
            }
        )

        row.update(
            regression_metrics(group)
        )

        row.update(
            classification_metrics(group)
        )

        row["forecast_min_utc"] = (
            group[
                "forecast_timestamp_utc"
            ].min().isoformat()
        )

        row["forecast_max_utc"] = (
            group[
                "forecast_timestamp_utc"
            ].max().isoformat()
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "forecast_max_utc"
    ).reset_index(drop=True)


def merge_evaluation_results(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if existing_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing_df.copy()
    else:
        combined = pd.concat(
            [
                existing_df,
                new_df,
            ],
            ignore_index=True,
            sort=False,
        )

    key_columns = [
        column
        for column in EVALUATION_KEY_COLUMNS
        if column in combined.columns
    ]

    if not key_columns:
        raise ValueError(
            "Cannot deduplicate evaluation results: "
            "no evaluation key columns are present."
        )

    for column in [
        "forecast_timestamp_utc",
        "evaluated_at_utc",
        "timestamp_utc",
        "prediction_ready_utc",
        "actual_timestamp_utc",
    ]:
        if column in combined.columns:
            combined[column] = pd.to_datetime(
                combined[column],
                utc=True,
                errors="coerce",
            )

    before_count = len(combined)

    combined = (
        combined
        .sort_values(
            [
                column
                for column in [
                    "evaluated_at_utc",
                    "forecast_timestamp_utc",
                ]
                if column in combined.columns
            ]
        )
        .drop_duplicates(
            subset=key_columns,
            keep="last",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = before_count - len(combined)

    return combined, duplicates_removed


def build_summary(
    eval_df: pd.DataFrame,
    split_name: str,
    combined_eval_df: pd.DataFrame | None = None,
    duplicates_removed: int = 0,
) -> dict:
    if combined_eval_df is None:
        combined_eval_df = eval_df

    overall_regression = (
        regression_metrics(eval_df)
    )

    overall_classification = (
        classification_metrics(eval_df)
    )

    return {
        "evaluated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": "ok",
        "split_name": split_name,
        "source_predictions_table": (
            PREDICTIONS_TABLE
        ),
        "source_actuals_table": (
            ACTUALS_TABLE
            if split_name == "production"
            else DATASET_SPLITS_TABLE
        ),
        "actual_match_tolerance_minutes": (
            ACTUAL_MATCH_TOLERANCE_MINUTES
            if split_name == "production"
            else None
        ),
        **overall_regression,
        **overall_classification,
        "stations_evaluated": int(
            eval_df[
                "station_name"
            ].nunique()
        ),
        "model_versions_evaluated": (
            int(
                eval_df[
                    "model_version"
                ].nunique()
            )
            if "model_version"
            in eval_df.columns
            else None
        ),
        "target_modes_evaluated": (
            sorted(
                eval_df[
                    "target_mode"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "target_mode"
            in eval_df.columns
            else None
        ),
        "horizons_evaluated": (
            sorted(
                pd.to_numeric(
                    eval_df[
                        "prediction_horizon_hours"
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if "prediction_horizon_hours"
            in eval_df.columns
            else None
        ),
        "forecast_min_utc": (
            eval_df[
                "forecast_timestamp_utc"
            ].min().isoformat()
        ),
        "forecast_max_utc": (
            eval_df[
                "forecast_timestamp_utc"
            ].max().isoformat()
        ),
        "new_evaluation_rows": int(
            len(eval_df)
        ),
        "total_evaluation_rows_after_merge": int(
            len(combined_eval_df)
        ),
        "duplicate_rows_removed": int(
            duplicates_removed
        ),
        "evaluation_key_columns": (
            EVALUATION_KEY_COLUMNS
        ),
        "threshold_config_path": str(THRESHOLD_CONFIG_PATH),
        "stations_with_thresholds": sorted(THRESHOLD_BY_STATION.keys()),
    }


def main():
    split_name = EVAL_SPLIT_NAME

    pred_df = load_predictions()

    if split_name == "production":
        actual_df = load_production_actuals()
    else:
        actual_df = load_split_actuals()

    try:
        eval_df = prepare_evaluation_frame(
            pred_df,
            actual_df,
            split_name,
        )

    except RuntimeError as exc:
        split_pred_df = pred_df[
            pred_df["split_name"]
            == split_name
        ].copy()

        target_column = None

        if (
            "target_column"
            in split_pred_df.columns
            and not split_pred_df.empty
        ):
            target_column = str(
                split_pred_df[
                    "target_column"
                ].iloc[0]
            )

        summary = build_not_ready_summary(
            split_pred_df,
            str(exc),
            split_name,
            target_column,
        )

        with open(
            EVALUATION_SUMMARY_JSON,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
            )

        print(
            "=== Gauge 24h evaluation "
            "not ready ==="
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print(
            f"\nWrote: "
            f"{EVALUATION_SUMMARY_JSON}"
        )

        sys.exit(0)

    try:
        existing_eval_df = load_bigquery_table(
            EVALUATION_TABLE,
            dataset=CURATED_DATASET,
            order_by=(
                "split_name, "
                "forecast_timestamp_utc, "
                "station_name"
            ),
        )
    except Exception:
        existing_eval_df = pd.DataFrame()

    combined_eval_df, duplicates_removed = (
        merge_evaluation_results(
            existing_eval_df,
            eval_df,
        )
    )

    write_bigquery_table(
        combined_eval_df,
        EVALUATION_TABLE,
        dataset=CURATED_DATASET,
        if_exists="replace",
    )

    summary = build_summary(
        eval_df=eval_df,
        split_name=split_name,
        combined_eval_df=combined_eval_df,
        duplicates_removed=duplicates_removed,
    )

    station_metrics_df = (
        build_station_metrics(eval_df)
    )

    model_metrics_df = (
        build_model_metrics(
            eval_df,
            split_name,
        )
    )

    eval_df.to_csv(
        EVALUATION_CSV,
        index=False,
    )

    station_metrics_df.to_csv(
        STATION_METRICS_CSV,
        index=False,
    )

    model_metrics_df.to_csv(
        MODEL_METRICS_CSV,
        index=False,
    )

    with open(
        EVALUATION_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "=== Gauge 24h evaluation "
        "summary ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nWorst stations by RMSE:"
    )

    print(
        station_metrics_df
        .head(10)
        .to_string(index=False)
    )

    print(
        f"\nWrote: {EVALUATION_CSV}"
    )

    print(
        f"Wrote: "
        f"{STATION_METRICS_CSV}"
    )

    print(
        f"Wrote: "
        f"{MODEL_METRICS_CSV}"
    )

    print(
        "Wrote BigQuery table: "
        f"{CURATED_DATASET}."
        f"{EVALUATION_TABLE}"
    )

    print(
        "Wrote: "
        f"{EVALUATION_SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()