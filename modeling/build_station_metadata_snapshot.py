from pathlib import Path
import pandas as pd
import numpy as np

from modeling.data_loader import load_bigquery_table


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "supervised_gauge_24h_multisource"
TARGET_COLUMN = "target_value_t_plus_24h"


def main():
    df = load_bigquery_table(TABLE_NAME)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    numeric_cols = [
        "target_value",
        TARGET_COLUMN,
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
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["station_name"]

    agg_map = {
        "timeseries_name": lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None,
        "unit": lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None,
        "source": lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else None,
        "timestamp_utc": ["min", "max", "count"],
    }

    optional_aggs = {
        "distance_km": ["mean", "median"],
        "target_value": ["mean", "std", "min", "max", "median"],
        TARGET_COLUMN: ["mean", "std", "min", "max", "median"],
        "temperature_c": ["mean", "std"],
        "precipitation_mm": ["mean", "sum"],
        "wind_speed_ms": ["mean", "std"],
        "pressure_hpa": ["mean", "std"],
        "relative_humidity_pct": ["mean", "std"],
        "rolling_std_3": ["mean", "max"],
        "rolling_max_6": ["mean"],
        "rolling_min_6": ["mean"],
    }

    for col, agg in optional_aggs.items():
        if col in df.columns:
            agg_map[col] = agg

    station_df = df.groupby(group_cols, dropna=False).agg(agg_map)
    station_df.columns = [
        "_".join([c for c in col if c]).replace(f"{TARGET_COLUMN}_", "target_t_plus_24h_")
        for col in station_df.columns.to_flat_index()
    ]
    station_df = station_df.reset_index()

    if {"timestamp_utc_min", "timestamp_utc_max"}.issubset(station_df.columns):
        station_df["coverage_days"] = (
            pd.to_datetime(station_df["timestamp_utc_max"]) - pd.to_datetime(station_df["timestamp_utc_min"])
        ).dt.total_seconds() / 86400.0

    if {"target_value_max", "target_value_min"}.issubset(station_df.columns):
        station_df["target_value_range"] = station_df["target_value_max"] - station_df["target_value_min"]

    if {"target_t_plus_24h_max", "target_t_plus_24h_min"}.issubset(station_df.columns):
        station_df["target_t_plus_24h_range"] = (
            station_df["target_t_plus_24h_max"] - station_df["target_t_plus_24h_min"]
        )

    if "distance_km_mean" in station_df.columns:
        station_df["corridor_position"] = station_df["distance_km_mean"]

    station_df = station_df.sort_values("station_name").reset_index(drop=True)
    station_df.to_csv(OUTPUT_DIR / "station_metadata_snapshot.csv", index=False)

    preview_cols = [c for c in [
        "station_name",
        "timeseries_name_<lambda>",
        "source_<lambda>",
        "distance_km_mean",
        "corridor_position",
        "target_value_mean",
        "target_value_std",
        "target_value_range",
        "coverage_days",
        "timestamp_utc_count",
    ] if c in station_df.columns]

    print(station_df[preview_cols].head(20).to_string(index=False))
    print(f"\nWrote: {OUTPUT_DIR / 'station_metadata_snapshot.csv'}")
    print(f"Rows: {len(station_df)}")


if __name__ == "__main__":
    main()