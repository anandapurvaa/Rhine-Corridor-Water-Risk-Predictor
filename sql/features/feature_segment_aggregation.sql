CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.feature_segment_aggregation` AS
SELECT
  station_id AS segment_id,
  timestamp_utc,
  MIN(target_value) AS segment_min_waterlevel,
  MAX(target_value) AS segment_max_waterlevel,
  AVG(target_value) AS segment_avg_waterlevel,
  COUNT(*) AS segment_count
FROM `rhine-corridor-navigator.rhein_curated.feature_modeling_multisource_v2`
WHERE target_value IS NOT NULL
  AND station_id IS NOT NULL
GROUP BY station_id, timestamp_utc;