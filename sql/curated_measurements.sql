CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.pegelonline_measurements_curated` AS
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
FROM `rhine-corridor-navigator.rhein_raw.v_pegelonline_measurements_dedup`;