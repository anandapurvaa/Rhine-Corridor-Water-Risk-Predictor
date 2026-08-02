CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_gauge_72h` AS
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
    source_url,
    LEAD(value, 1) OVER w AS lead_1,
    LEAD(value, 3) OVER w AS lead_3,
    LEAD(value, 6) OVER w AS lead_6,
    LEAD(value, 12) OVER w AS lead_12,
    LEAD(value, 24) OVER w AS lead_24,
    LEAD(value, 36) OVER w AS lead_36,
    LEAD(value, 48) OVER w AS lead_48,
    LEAD(value, 60) OVER w AS lead_60,
    LEAD(value, 72) OVER w AS lead_72
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_timeseries`
  WHERE timeseries_name = 'W'
  WINDOW w AS (
    PARTITION BY station_name, timeseries_name
    ORDER BY timestamp_utc
  )
)
SELECT
  *,
  CASE
    WHEN GREATEST(
      COALESCE(lead_1 <= 120, FALSE),
      COALESCE(lead_3 <= 120, FALSE),
      COALESCE(lead_6 <= 120, FALSE),
      COALESCE(lead_12 <= 120, FALSE),
      COALESCE(lead_24 <= 120, FALSE),
      COALESCE(lead_36 <= 120, FALSE),
      COALESCE(lead_48 <= 120, FALSE),
      COALESCE(lead_60 <= 120, FALSE),
      COALESCE(lead_72 <= 120, FALSE)
    ) THEN 1 ELSE 0
  END AS target_low_water_72h
FROM base;