CREATE OR REPLACE VIEW `rhine-corridor-navigator.rhein_raw.v_pegelonline_measurements_dedup` AS
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
FROM `rhine-corridor-navigator.rhein_raw.pegelonline_measurements`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY station_id, timeseries_name, timestamp_utc
  ORDER BY ingestion_ts_utc DESC, source_record_hash DESC
) = 1;