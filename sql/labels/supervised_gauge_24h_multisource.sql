CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource` AS
WITH base AS (
  SELECT
    *,
    LEAD(target_value, 24) OVER (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
    ) AS target_value_t_plus_24h
  FROM `rhine-corridor-navigator.rhein_curated.feature_modeling_multisource_v2`
)
SELECT
  *,
  target_value_t_plus_24h - target_value AS target_delta_24h
FROM base
WHERE target_value_t_plus_24h IS NOT NULL;