CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_join` AS
WITH gauge_base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_timeseries`
  WHERE timeseries_name = 'W'
),
weather_base AS (
  SELECT
    dwd_station_id,
    dwd_station_name,
    timestamp_utc,
    temperature_c,
    precipitation_mm,
    wind_speed_ms,
    pressure_hpa,
    relative_humidity_pct
  FROM `rhine-corridor-navigator.rhein_raw.dwd_hourly_observations`
),
mapping AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.map_gauge_to_dwd_station`
)
SELECT
  g.*,
  m.dwd_station_id,
  m.dwd_station_name,
  m.distance_km,
  w.temperature_c,
  w.precipitation_mm,
  w.wind_speed_ms,
  w.pressure_hpa,
  w.relative_humidity_pct
FROM gauge_base g
LEFT JOIN mapping m
  ON g.station_id = m.station_id
LEFT JOIN weather_base w
  ON m.dwd_station_id = w.dwd_station_id
 AND TIMESTAMP_TRUNC(g.timestamp_utc, HOUR) = TIMESTAMP_TRUNC(w.timestamp_utc, HOUR);