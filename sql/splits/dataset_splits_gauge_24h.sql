CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h` AS
WITH params AS (
  SELECT 0.70 AS train_fraction, 0.15 AS validation_fraction, 24 * 7 AS gap_hours
),
all_rows AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource`
  WHERE timestamp_utc IS NOT NULL
),
labeled AS (
  SELECT * FROM all_rows
  WHERE target_value_t_plus_24h IS NOT NULL
),
timeline AS (
  SELECT timestamp_utc,
         ROW_NUMBER() OVER (ORDER BY timestamp_utc) AS rn,
         COUNT(*) OVER () AS n
  FROM (SELECT DISTINCT timestamp_utc FROM labeled)
),
cutoffs AS (
  SELECT
    MIN(CASE WHEN rn >= CAST(n * train_fraction AS INT64) THEN timestamp_utc END) AS train_end_utc,
    MIN(CASE WHEN rn >= CAST(n * (train_fraction + validation_fraction) AS INT64) THEN timestamp_utc END) AS validation_end_utc,
    MIN(gap_hours) AS gap_hours
  FROM timeline CROSS JOIN params
),
boundaries AS (
  SELECT
    train_end_utc,
    TIMESTAMP_ADD(train_end_utc, INTERVAL gap_hours HOUR) AS validation_start_utc,
    validation_end_utc,
    TIMESTAMP_ADD(validation_end_utc, INTERVAL gap_hours HOUR) AS test_start_utc
  FROM cutoffs
),
labeled_split AS (
  SELECT
    l.*,
    CASE
      WHEN l.timestamp_utc < b.train_end_utc THEN 'train'
      WHEN l.timestamp_utc >= b.validation_start_utc
       AND l.timestamp_utc < b.validation_end_utc THEN 'validation'
      WHEN l.timestamp_utc >= b.test_start_utc THEN 'test'
      ELSE 'gap'
    END AS split_name,
    CASE
      WHEN l.timestamp_utc >= b.train_end_utc
       AND l.timestamp_utc < b.validation_start_utc THEN 'train_validation_gap'
      WHEN l.timestamp_utc >= b.validation_end_utc
       AND l.timestamp_utc < b.test_start_utc THEN 'validation_test_gap'
      ELSE NULL
    END AS gap_reason,
    b.train_end_utc,
    b.validation_start_utc,
    b.validation_end_utc,
    b.test_start_utc
  FROM labeled l CROSS JOIN boundaries b
),
production_candidates AS (
  SELECT * EXCEPT(row_number)
  FROM (
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY a.station_name, a.timeseries_name
        ORDER BY a.timestamp_utc DESC
      ) AS row_number
    FROM all_rows a
    CROSS JOIN boundaries b
    WHERE a.timestamp_utc >= b.test_start_utc
      AND a.temperature_c IS NOT NULL
      AND a.precipitation_mm IS NOT NULL
      AND a.wind_speed_ms IS NOT NULL
      AND a.pressure_hpa IS NOT NULL
      AND a.relative_humidity_pct IS NOT NULL
  )
  WHERE row_number = 1
),
production_split AS (
  SELECT
    p.*,
    'production' AS split_name,
    CAST(NULL AS STRING) AS gap_reason,
    b.train_end_utc,
    b.validation_start_utc,
    b.validation_end_utc,
    b.test_start_utc
  FROM production_candidates p CROSS JOIN boundaries b
)
SELECT * FROM labeled_split
UNION ALL BY NAME
SELECT * FROM production_split;