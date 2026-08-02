CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h` AS
WITH base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource`
  WHERE target_value_t_plus_24h IS NOT NULL
),
numbered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY station_id, timeseries_name
      ORDER BY timestamp_utc
    ) AS rn,
    COUNT(*) OVER (
      PARTITION BY station_id, timeseries_name
    ) AS n
  FROM base
)
SELECT
  *,
  CASE
    WHEN rn <= CAST(n * 0.70 AS INT64) THEN 'train'
    WHEN rn <= CAST(n * 0.85 AS INT64) THEN 'validation'
    ELSE 'test'
  END AS split_name
FROM numbered;