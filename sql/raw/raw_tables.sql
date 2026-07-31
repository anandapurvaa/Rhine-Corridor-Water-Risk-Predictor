CREATE SCHEMA IF NOT EXISTS `rhein_raw`;

CREATE TABLE IF NOT EXISTS `rhein_raw.pegelonline_measurements` (
  station_id STRING,
  station_name STRING,
  timeseries_name STRING,
  timestamp_utc TIMESTAMP,
  value FLOAT64,
  unit STRING,
  latitude FLOAT64,
  longitude FLOAT64,
  ingestion_ts_utc TIMESTAMP,
  source STRING,
  source_record_hash STRING,
  source_url STRING
)
PARTITION BY DATE(ingestion_ts_utc)
CLUSTER BY station_name, timeseries_name, source_record_hash;