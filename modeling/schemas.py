from __future__ import annotations

TABLE_NAME = "supervised_gauge_24h_multisource"
TARGET_COLUMN = "target_value_t_plus_24h"

CATEGORICAL_COLUMNS = [
    "station_name",
    "timeseries_name",
    "unit",
    "source",
]

NUMERIC_COLUMNS_LEAN = [
    "target_value",
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

PRODUCTION_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS_LEAN

OPTIONAL_METADATA_COLUMNS = [
    "station_id",
    "latitude",
    "longitude",
]