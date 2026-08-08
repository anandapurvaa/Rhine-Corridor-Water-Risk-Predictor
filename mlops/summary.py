from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUMMARY_KEYS = {
    "rows_ingested",
    "rows_predicted",
    "stations_processed",
    "stations_failed",
    "files_executed",
    "files_failed",
    "model_version",
    "prediction_table",
    "prediction_split",
    "latest_input_timestamp",
    "input_age_hours",
    "watermark_before",
    "watermark_after",
    "source",
    "mode",
    "data_quality_status",
    "quality_metrics_written",
    "quality_failed_metrics",
}


def normalize_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return {
            str(key): val
            for key, val in value.items()
        }

    return {}


def apply_summary(
    target: dict[str, Any],
    summary: Mapping[str, Any] | None,
) -> None:
    if not summary:
        return

    for key, value in summary.items():
        if key in SUMMARY_KEYS:
            target[key] = value