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
    TIMESTAMP_TRUNC(timestamp_utc, HOUR) AS timestamp_hour_utc,
    temperature_c,
    precipitation_mm,
    wind_speed_ms,
    pressure_hpa,
    relative_humidity_pct
  FROM `rhine-corridor-navigator.rhein_raw.dwd_hourly_observations`
),
mapping AS (
  SELECT
    station_id,
    station_name,
    primary_dwd_station_id,
    primary_dwd_station_name,
    primary_distance_km,
    primary_blend_weight,
    backup1_dwd_station_id,
    backup1_dwd_station_name,
    backup1_distance_km,
    backup1_blend_weight,
    backup2_dwd_station_id,
    backup2_dwd_station_name,
    backup2_distance_km,
    backup2_blend_weight,
    matched_station_count
  FROM `rhine-corridor-navigator.rhein_curated.map_gauge_to_dwd_station`
),
joined AS (
  SELECT
    g.*,

    m.primary_dwd_station_id,
    m.primary_dwd_station_name,
    m.primary_distance_km,
    m.primary_blend_weight,
    m.backup1_dwd_station_id,
    m.backup1_dwd_station_name,
    m.backup1_distance_km,
    m.backup1_blend_weight,
    m.backup2_dwd_station_id,
    m.backup2_dwd_station_name,
    m.backup2_distance_km,
    m.backup2_blend_weight,
    m.matched_station_count,

    wp.dwd_station_id AS matched_dwd_station_id,
    wp.dwd_station_name AS matched_dwd_station_name,
    wp.timestamp_hour_utc AS weather_timestamp_utc,
    wp.temperature_c AS p_temperature_c,
    wp.precipitation_mm AS p_precipitation_mm,
    wp.wind_speed_ms AS p_wind_speed_ms,
    wp.pressure_hpa AS p_pressure_hpa,
    wp.relative_humidity_pct AS p_relative_humidity_pct,

    wb1.temperature_c AS b1_temperature_c,
    wb1.precipitation_mm AS b1_precipitation_mm,
    wb1.wind_speed_ms AS b1_wind_speed_ms,
    wb1.pressure_hpa AS b1_pressure_hpa,
    wb1.relative_humidity_pct AS b1_relative_humidity_pct,

    wb2.temperature_c AS b2_temperature_c,
    wb2.precipitation_mm AS b2_precipitation_mm,
    wb2.wind_speed_ms AS b2_wind_speed_ms,
    wb2.pressure_hpa AS b2_pressure_hpa,
    wb2.relative_humidity_pct AS b2_relative_humidity_pct
  FROM gauge_base g
  LEFT JOIN mapping m
    ON g.station_id = m.station_id
  LEFT JOIN weather_base wp
    ON wp.dwd_station_id = m.primary_dwd_station_id
   AND wp.timestamp_hour_utc = TIMESTAMP_TRUNC(g.timestamp_utc, HOUR)
  LEFT JOIN weather_base wb1
    ON wb1.dwd_station_id = m.backup1_dwd_station_id
   AND wb1.timestamp_hour_utc = TIMESTAMP_TRUNC(g.timestamp_utc, HOUR)
  LEFT JOIN weather_base wb2
    ON wb2.dwd_station_id = m.backup2_dwd_station_id
   AND wb2.timestamp_hour_utc = TIMESTAMP_TRUNC(g.timestamp_utc, HOUR)
)
SELECT
  *,
  primary_dwd_station_id AS dwd_station_id,
  primary_dwd_station_name AS dwd_station_name,
  primary_distance_km AS distance_km,

  COALESCE(p_temperature_c, b1_temperature_c, b2_temperature_c) AS temperature_c,
  COALESCE(p_precipitation_mm, b1_precipitation_mm, b2_precipitation_mm) AS precipitation_mm,
  COALESCE(p_wind_speed_ms, b1_wind_speed_ms, b2_wind_speed_ms) AS wind_speed_ms,
  COALESCE(p_pressure_hpa, b1_pressure_hpa, b2_pressure_hpa) AS pressure_hpa,
  COALESCE(p_relative_humidity_pct, b1_relative_humidity_pct, b2_relative_humidity_pct) AS relative_humidity_pct,

  COALESCE(p_temperature_c, b1_temperature_c, b2_temperature_c) AS temperature_c_blend,
  COALESCE(p_precipitation_mm, b1_precipitation_mm, b2_precipitation_mm) AS precipitation_mm_blend,
  COALESCE(p_wind_speed_ms, b1_wind_speed_ms, b2_wind_speed_ms) AS wind_speed_ms_blend,
  COALESCE(p_pressure_hpa, b1_pressure_hpa, b2_pressure_hpa) AS pressure_hpa_blend,
  COALESCE(p_relative_humidity_pct, b1_relative_humidity_pct, b2_relative_humidity_pct) AS relative_humidity_pct_blend,

  CASE
    WHEN p_temperature_c IS NOT NULL THEN 'primary'
    WHEN b1_temperature_c IS NOT NULL THEN 'backup1'
    WHEN b2_temperature_c IS NOT NULL THEN 'backup2'
    ELSE NULL
  END AS weather_source_used
FROM joined;