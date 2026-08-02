CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_segment_24h` AS
WITH base AS (
  SELECT
    segment_id,
    timestamp_utc,
    segment_min_waterlevel,
    segment_max_waterlevel,
    segment_avg_waterlevel,
    segment_count,
    CASE
      WHEN segment_id = 'KAUB' AND segment_min_waterlevel <= 120 THEN 1
      WHEN segment_id = 'MAXAU' AND segment_min_waterlevel <= 380 THEN 1
      WHEN segment_id = 'KOBLENZ' AND segment_min_waterlevel <= 150 THEN 1
      WHEN segment_id = 'DUISBURG-RUHRORT' AND segment_min_waterlevel <= 260 THEN 1
      WHEN segment_id = 'EMMERICH' AND segment_min_waterlevel <= 140 THEN 1
      WHEN segment_id = 'KÖLN' AND segment_min_waterlevel <= 180 THEN 1
      WHEN segment_id = 'MAINZ' AND segment_min_waterlevel <= 170 THEN 1
      WHEN segment_id = 'WORMS' AND segment_min_waterlevel <= 120 THEN 1
      WHEN segment_id = 'SPEYER' AND segment_min_waterlevel <= 200 THEN 1
      WHEN segment_id = 'BONN' AND segment_min_waterlevel <= 170 THEN 1
      WHEN segment_id = 'DÜSSELDORF' AND segment_min_waterlevel <= 190 THEN 1
      WHEN segment_id = 'REES' AND segment_min_waterlevel <= 160 THEN 1
      ELSE 0
    END AS target_low_water_24h
  FROM `rhine-corridor-navigator.rhein_curated.feature_segment_aggregation`
)
SELECT * FROM base;