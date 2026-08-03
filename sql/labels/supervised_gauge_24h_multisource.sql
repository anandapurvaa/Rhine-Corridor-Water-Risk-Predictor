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

    dwd_station_id,
    dwd_station_name,
    distance_km,

    primary_dwd_station_id,
    primary_dwd_station_name,
    primary_distance_km,
    backup1_dwd_station_id,
    backup1_dwd_station_name,
    backup1_distance_km,
    backup2_dwd_station_id,
    backup2_dwd_station_name,
    backup2_distance_km,
    matched_station_count,
    weather_source_used,

    temperature_c,
    precipitation_mm,
    wind_speed_ms,
    pressure_hpa,
    relative_humidity_pct,

    temperature_c_blend,
    precipitation_mm_blend,
    wind_speed_ms_blend,
    pressure_hpa_blend,
    relative_humidity_pct_blend
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_enriched`
),
feat AS (
  SELECT
    *,
    LAG(target_value, 1) OVER w AS lag_1,
    LAG(target_value, 3) OVER w AS lag_3,
    LAG(target_value, 6) OVER w AS lag_6,

    target_value - LAG(target_value, 1) OVER w AS diff_1,
    target_value - LAG(target_value, 3) OVER w AS diff_3,

    AVG(target_value) OVER w3 AS rolling_mean_3,
    STDDEV(target_value) OVER w3 AS rolling_std_3,
    MIN(target_value) OVER w6 AS rolling_min_6,
    MAX(target_value) OVER w6 AS rolling_max_6,

    EXTRACT(HOUR FROM timestamp_utc) AS hour_utc,
    EXTRACT(DAYOFWEEK FROM timestamp_utc) AS day_of_week,
    EXTRACT(MONTH FROM timestamp_utc) AS month,

    LAG(temperature_c, 1) OVER w AS temp_lag_1,
    LAG(temperature_c, 3) OVER w AS temp_lag_3,
    LAG(temperature_c, 6) OVER w AS temp_lag_6,
    LAG(temperature_c, 12) OVER w AS temp_lag_12,

    LAG(precipitation_mm, 1) OVER w AS precip_lag_1,
    LAG(precipitation_mm, 3) OVER w AS precip_lag_3,
    LAG(precipitation_mm, 6) OVER w AS precip_lag_6,
    LAG(precipitation_mm, 12) OVER w AS precip_lag_12,

    LAG(wind_speed_ms, 1) OVER w AS wind_lag_1,
    LAG(wind_speed_ms, 3) OVER w AS wind_lag_3,
    LAG(wind_speed_ms, 6) OVER w AS wind_lag_6,

    LAG(pressure_hpa, 1) OVER w AS pressure_lag_1,
    LAG(pressure_hpa, 3) OVER w AS pressure_lag_3,

    LAG(relative_humidity_pct, 1) OVER w AS humidity_lag_1,

    AVG(temperature_c) OVER w3 AS temp_roll_mean_3,
    AVG(temperature_c) OVER w6 AS temp_roll_mean_6,
    AVG(temperature_c) OVER w12 AS temp_roll_mean_12,

    AVG(precipitation_mm) OVER w6 AS precip_roll_mean_6,
    AVG(precipitation_mm) OVER w12 AS precip_roll_mean_12,
    SUM(precipitation_mm) OVER w6 AS precip_roll_sum_6,
    SUM(precipitation_mm) OVER w12 AS precip_roll_sum_12,
    SUM(precipitation_mm) OVER w24 AS precip_roll_sum_24,

    AVG(wind_speed_ms) OVER w6 AS wind_roll_mean_6,
    AVG(wind_speed_ms) OVER w12 AS wind_roll_mean_12,

    AVG(pressure_hpa) OVER w6 AS pressure_roll_mean_6,
    AVG(pressure_hpa) OVER w12 AS pressure_roll_mean_12,

    AVG(relative_humidity_pct) OVER w6 AS humidity_roll_mean_6,
    AVG(relative_humidity_pct) OVER w12 AS humidity_roll_mean_12,

    pressure_hpa - LAG(pressure_hpa, 1) OVER w AS pressure_delta_1,
    pressure_hpa - LAG(pressure_hpa, 3) OVER w AS pressure_delta_3,
    LAG(temperature_c, 1) OVER w - LAG(temperature_c, 3) OVER w AS temp_change_1_3,
    SUM(precipitation_mm) OVER w12 - SUM(precipitation_mm) OVER w24 AS precip_accel_12_24,

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
    ),
    w12 AS (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
      ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
    ),
    w24 AS (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
      ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    )
)
SELECT *
FROM feat;