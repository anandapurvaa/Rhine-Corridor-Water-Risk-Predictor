CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_segment_24h` AS
WITH base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.feature_segment_aggregation`
),
future_labels AS (
  SELECT
    a.segment_id,
    a.timestamp_utc,
    MAX(CASE WHEN b.segment_min_waterlevel IS NOT NULL AND b.segment_min_waterlevel <= 150 THEN 1 ELSE 0 END) AS target_low_water_24h
  FROM base a
  LEFT JOIN base b
    ON a.segment_id = b.segment_id
   AND b.timestamp_utc > a.timestamp_utc
   AND b.timestamp_utc <= TIMESTAMP_ADD(a.timestamp_utc, INTERVAL 24 HOUR)
  GROUP BY a.segment_id, a.timestamp_utc
)
SELECT
  a.*,
  f.target_low_water_24h
FROM base a
LEFT JOIN future_labels f
  ON a.segment_id = f.segment_id
 AND a.timestamp_utc = f.timestamp_utc;