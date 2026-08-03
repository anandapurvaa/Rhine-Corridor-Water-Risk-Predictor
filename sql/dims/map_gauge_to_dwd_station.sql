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
  WHERE ST_DWITHIN(
    ST_GEOGPOINT(g.longitude, g.latitude),
    ST_GEOGPOINT(d.longitude, d.latitude),
    250000
  )
),
scored AS (
  SELECT
    *,
    CASE
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'KÖLN|KOELN|COLOGNE')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'KÖLN|KOELN|COLOGNE') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'MAINZ')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'MAINZ') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'MANNHEIM')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'MANNHEIM') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'KOBLENZ')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'KOBLENZ|BENDORF') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'BONN')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'BONN|KÖNIGSWINTER|KOENIGSWINTER') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'DUISBURG|RUHRORT')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'DUISBURG|ESSEN') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'EMMERICH|REES|WESEL')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'WESEL|EMMERICH|DÜSSELDORF|DUESSELDORF') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'BASEL|RHEINWEILER|BREISACH|RUST|OTTENHEIM|KEHL|IFFEZHEIM|MAXAU|PHILIPPSBURG|PLITTERSDORF|SPEYER|WORMS')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'KARLSRUHE|MANNHEIM|FREIBURG|OFFENBURG|BASEL|KONSTANZ') THEN 1
      WHEN REGEXP_CONTAINS(UPPER(station_name), r'KONSTANZ')
           AND REGEXP_CONTAINS(UPPER(dwd_station_name), r'KONSTANZ') THEN 1
      ELSE 0
    END AS name_bonus
  FROM pairs
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY station_id
      ORDER BY name_bonus DESC, distance_km ASC, dwd_station_id ASC
    ) AS station_rank
  FROM scored
),
top3 AS (
  SELECT *
  FROM ranked
  WHERE station_rank <= 3
),
weighted AS (
  SELECT
    station_id,
    station_name,
    dwd_station_id,
    dwd_station_name,
    distance_km,
    name_bonus,
    station_rank,
    CASE
      WHEN distance_km = 0 THEN 1000000000.0
      ELSE 1.0 / POW(distance_km, 2)
    END AS raw_weight
  FROM top3
),
normalized AS (
  SELECT
    *,
    SAFE_DIVIDE(
      raw_weight,
      SUM(raw_weight) OVER (PARTITION BY station_id)
    ) AS blend_weight
  FROM weighted
),
pivoted AS (
  SELECT
    station_id,
    ANY_VALUE(station_name) AS station_name,

    MAX(IF(station_rank = 1, dwd_station_id, NULL)) AS primary_dwd_station_id,
    MAX(IF(station_rank = 1, dwd_station_name, NULL)) AS primary_dwd_station_name,
    MAX(IF(station_rank = 1, distance_km, NULL)) AS primary_distance_km,
    MAX(IF(station_rank = 1, name_bonus, NULL)) AS primary_name_bonus,
    MAX(IF(station_rank = 1, blend_weight, NULL)) AS primary_blend_weight,

    MAX(IF(station_rank = 2, dwd_station_id, NULL)) AS backup1_dwd_station_id,
    MAX(IF(station_rank = 2, dwd_station_name, NULL)) AS backup1_dwd_station_name,
    MAX(IF(station_rank = 2, distance_km, NULL)) AS backup1_distance_km,
    MAX(IF(station_rank = 2, name_bonus, NULL)) AS backup1_name_bonus,
    MAX(IF(station_rank = 2, blend_weight, NULL)) AS backup1_blend_weight,

    MAX(IF(station_rank = 3, dwd_station_id, NULL)) AS backup2_dwd_station_id,
    MAX(IF(station_rank = 3, dwd_station_name, NULL)) AS backup2_dwd_station_name,
    MAX(IF(station_rank = 3, distance_km, NULL)) AS backup2_distance_km,
    MAX(IF(station_rank = 3, name_bonus, NULL)) AS backup2_name_bonus,
    MAX(IF(station_rank = 3, blend_weight, NULL)) AS backup2_blend_weight,

    COUNT(*) AS matched_station_count
  FROM normalized
  GROUP BY station_id
)
SELECT
  station_id,
  station_name,

  primary_dwd_station_id,
  primary_dwd_station_name,
  primary_distance_km,
  primary_name_bonus,
  primary_blend_weight,

  backup1_dwd_station_id,
  backup1_dwd_station_name,
  backup1_distance_km,
  backup1_name_bonus,
  backup1_blend_weight,

  backup2_dwd_station_id,
  backup2_dwd_station_name,
  backup2_distance_km,
  backup2_name_bonus,
  backup2_blend_weight,

  matched_station_count
FROM pivoted;