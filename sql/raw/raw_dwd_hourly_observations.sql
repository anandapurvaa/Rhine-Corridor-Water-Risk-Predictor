CREATE TABLE IF NOT EXISTS `rhine-corridor-navigator.rhein_raw.dwd_hourly_observations` (
  dwd_station_id STRING,
  dwd_station_name STRING,
  timestamp_utc TIMESTAMP,
  latitude FLOAT64,
  longitude FLOAT64,
  temperature_c FLOAT64,
  precipitation_mm FLOAT64,
  wind_speed_ms FLOAT64,
  pressure_hpa FLOAT64,
  relative_humidity_pct FLOAT64,
  is_proxy_backfilled BOOL,
  proxy_source_station_id STRING,
  proxy_source_variable STRING,
  proxy_source_distance_km FLOAT64,
  proxy_fill_method STRING,
  ingestion_ts_utc TIMESTAMP,
  source STRING,
  source_record_hash STRING,
  source_url STRING
)
PARTITION BY DATE(ingestion_ts_utc)
CLUSTER BY dwd_station_id, source_record_hash;