CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_enriched` AS
WITH base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_join`
),
feat AS (
  SELECT
    *,
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
    pressure_hpa - LAG(pressure_hpa, 3) OVER w AS pressure_delta_3
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
SELECT
  *,
  temp_lag_1 - temp_lag_3 AS temp_change_1_3,
  precip_roll_sum_24 - precip_roll_sum_12 AS precip_accel_12_24
FROM feat;