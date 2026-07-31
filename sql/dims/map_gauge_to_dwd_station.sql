CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.map_gauge_to_dwd_station` AS
WITH dwd_stations AS (
  SELECT DISTINCT
    dwd_station_id,
    dwd_station_name,
    latitude,
    longitude
  FROM `rhine-corridor-navigator.rhein_raw.dwd_hourly_observations`
  WHERE latitude IS NOT NULL
    AND longitude IS NOT NULL
),
gauge_stations AS (
  SELECT DISTINCT
    station_id,
    station_name,
    latitude,
    longitude
  FROM `rhine-corridor-navigator.rhein_curated.dim_station`
  WHERE latitude IS NOT NULL
    AND longitude IS NOT NULL
),
pairs AS (
  SELECT
    g.station_id,
    g.station_name,
    d.dwd_station_id,
    d.dwd_station_name,
    ST_DISTANCE(
      ST_GEOGPOINT(g.longitude, g.latitude),
      ST_GEOGPOINT(d.longitude, d.latitude)
    ) / 1000.0 AS distance_km
  FROM gauge_stations g
  CROSS JOIN dwd_stations d
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY distance_km ASC) AS rn
  FROM pairs
)
SELECT
  station_id,
  station_name,
  dwd_station_id,
  dwd_station_name,
  distance_km
FROM ranked
WHERE rn = 1;