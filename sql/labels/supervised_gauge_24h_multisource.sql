CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource` AS
WITH base AS (
  SELECT
    station_id,
    station_name,
    timeseries_name,
    timestamp_utc,
    value AS target_value,
    unit,
    latitude,
    longitude,
    source,
    temperature_c,
    precipitation_mm,
    wind_speed_ms,
    pressure_hpa,
    relative_humidity_pct,
    temp_lag_1,
    temp_lag_3,
    temp_lag_6,
    temp_lag_12,
    precip_lag_1,
    precip_lag_3,
    precip_lag_6,
    precip_lag_12,
    wind_lag_1,
    wind_lag_3,
    wind_lag_6,
    pressure_lag_1,
    pressure_lag_3,
    humidity_lag_1,
    temp_roll_mean_3,
    temp_roll_mean_6,
    temp_roll_mean_12,
    precip_roll_mean_6,
    precip_roll_mean_12,
    precip_roll_sum_6,
    precip_roll_sum_12,
    precip_roll_sum_24,
    wind_roll_mean_6,
    wind_roll_mean_12,
    pressure_roll_mean_6,
    pressure_roll_mean_12,
    humidity_roll_mean_6,
    humidity_roll_mean_12,
    pressure_delta_1,
    pressure_delta_3,
    temp_change_1_3,
    precip_accel_12_24
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_enriched`
),
feat AS (
  SELECT
    *,
    LAG(target_value, 1) OVER w AS lag_1,
    LAG(target_value, 3) OVER w AS lag_3,
    LAG(target_value, 6) OVER w AS lag_6,
    AVG(target_value) OVER w3 AS rolling_mean_3,
    STDDEV(target_value) OVER w3 AS rolling_std_3,
    MIN(target_value) OVER w6 AS rolling_min_6,
    MAX(target_value) OVER w6 AS rolling_max_6,
    EXTRACT(HOUR FROM timestamp_utc) AS hour_utc,
    EXTRACT(DAYOFWEEK FROM timestamp_utc) AS day_of_week,
    EXTRACT(MONTH FROM timestamp_utc) AS month,
    LEAD(target_value, 24) OVER w AS target_value_t_plus_24h
  FROM base
  WINDOW
    w AS (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
    ),
    w3 AS (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ),
    w6 AS (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
      ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
    )
)
SELECT *
FROM feat;