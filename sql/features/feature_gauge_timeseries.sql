CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.feature_gauge_timeseries` AS
WITH base AS (
  SELECT
    station_id,
    station_name,
    timeseries_name,
    timestamp_utc,
    value,
    unit,
    latitude,
    longitude,
    ingestion_ts_utc,
    source,
    source_record_hash,
    source_url
  FROM `rhine-corridor-navigator.rhein_curated.pegelonline_measurements_curated`
  WHERE value IS NOT NULL
)
SELECT
  *,
  LAG(value, 1) OVER w AS lag_1,
  LAG(value, 3) OVER w AS lag_3,
  LAG(value, 6) OVER w AS lag_6,
  value - LAG(value, 1) OVER w AS diff_1,
  value - LAG(value, 3) OVER w AS diff_3,
  AVG(value) OVER w3 AS rolling_mean_3,
  STDDEV(value) OVER w3 AS rolling_std_3,
  MIN(value) OVER w6 AS rolling_min_6,
  MAX(value) OVER w6 AS rolling_max_6,
  EXTRACT(HOUR FROM timestamp_utc) AS hour_utc,
  EXTRACT(DAYOFWEEK FROM timestamp_utc) AS day_of_week,
  EXTRACT(MONTH FROM timestamp_utc) AS month
FROM base
WINDOW
  w AS (
    PARTITION BY station_name, timeseries_name
    ORDER BY timestamp_utc
  ),
  w3 AS (
    PARTITION BY station_name, timeseries_name
    ORDER BY timestamp_utc
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
  ),
  w6 AS (
    PARTITION BY station_name, timeseries_name
    ORDER BY timestamp_utc
    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
  );