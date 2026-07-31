CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h` AS
WITH base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource`
  WHERE target_value_t_plus_24h IS NOT NULL
),
ranked AS (
  SELECT
    *,
    PERCENT_RANK() OVER (ORDER BY timestamp_utc) AS pr
  FROM base
)
SELECT
  *,
  CASE
    WHEN pr < 0.70 THEN 'train'
    WHEN pr < 0.85 THEN 'validation'
    ELSE 'test'
  END AS split_name
FROM ranked;