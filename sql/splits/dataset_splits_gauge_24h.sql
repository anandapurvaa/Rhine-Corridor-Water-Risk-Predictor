CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h` AS
WITH params AS (
  SELECT
    0.70 AS train_fraction,
    0.15 AS validation_fraction,
    24 * 7 AS gap_hours
),
base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource`
  WHERE target_value_t_plus_24h IS NOT NULL
    AND timestamp_utc IS NOT NULL
),
timeline AS (
  SELECT
    timestamp_utc,
    ROW_NUMBER() OVER (ORDER BY timestamp_utc) AS rn,
    COUNT(*) OVER () AS n
  FROM (
    SELECT DISTINCT timestamp_utc
    FROM base
  )
),
cutoffs AS (
  SELECT
    MIN(CASE WHEN rn >= CAST(n * train_fraction AS INT64) THEN timestamp_utc END) AS train_end_utc,
    MIN(CASE WHEN rn >= CAST(n * (train_fraction + validation_fraction) AS INT64) THEN timestamp_utc END) AS validation_end_utc,
    MIN(gap_hours) AS gap_hours
  FROM timeline
  CROSS JOIN params
),
labeled AS (
  SELECT
    b.*,
    c.train_end_utc,
    TIMESTAMP_ADD(c.train_end_utc, INTERVAL c.gap_hours HOUR) AS validation_start_utc,
    c.validation_end_utc,
    TIMESTAMP_ADD(c.validation_end_utc, INTERVAL c.gap_hours HOUR) AS test_start_utc
  FROM base b
  CROSS JOIN cutoffs c
)
SELECT
  *,
  CASE
    WHEN timestamp_utc < train_end_utc THEN 'train'
    WHEN timestamp_utc >= validation_start_utc
         AND timestamp_utc < validation_end_utc THEN 'validation'
    WHEN timestamp_utc >= test_start_utc THEN 'test'
    ELSE 'gap'
  END AS split_name,
  CASE
    WHEN timestamp_utc >= train_end_utc
         AND timestamp_utc < validation_start_utc THEN 'train_validation_gap'
    WHEN timestamp_utc >= validation_end_utc
         AND timestamp_utc < test_start_utc THEN 'validation_test_gap'
    ELSE NULL
  END AS gap_reason
FROM labeled;