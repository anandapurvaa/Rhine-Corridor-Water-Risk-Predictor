CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dim_station` AS
SELECT DISTINCT
  station_id,
  station_name,
  latitude,
  longitude
FROM `rhine-corridor-navigator.rhein_raw.v_pegelonline_measurements_dedup`;