CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.supervised_gauge_72h` AS
WITH base AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_timeseries`
  WHERE timeseries_name = 'W'
),
future_labels AS (
  SELECT
    a.station_name,
    a.timestamp_utc,
    MAX(
      CASE
        WHEN b.station_name = 'KAUB' AND b.value <= 120 THEN 1
        WHEN b.station_name = 'MAXAU' AND b.value <= 380 THEN 1
        WHEN b.station_name = 'KOBLENZ' AND b.value <= 150 THEN 1
        WHEN b.station_name = 'DUISBURG-RUHRORT' AND b.value <= 260 THEN 1
        WHEN b.station_name = 'EMMERICH' AND b.value <= 140 THEN 1
        WHEN b.station_name = 'KÖLN' AND b.value <= 180 THEN 1
        WHEN b.station_name = 'MAINZ' AND b.value <= 170 THEN 1
        WHEN b.station_name = 'WORMS' AND b.value <= 120 THEN 1
        WHEN b.station_name = 'SPEYER' AND b.value <= 200 THEN 1
        WHEN b.station_name = 'BONN' AND b.value <= 170 THEN 1
        WHEN b.station_name = 'DÜSSELDORF' AND b.value <= 190 THEN 1
        WHEN b.station_name = 'REES' AND b.value <= 160 THEN 1
        ELSE 0
      END
    ) AS target_low_water_72h
  FROM base a
  LEFT JOIN base b
    ON a.station_name = b.station_name
   AND b.timestamp_utc > a.timestamp_utc
   AND b.timestamp_utc <= TIMESTAMP_ADD(a.timestamp_utc, INTERVAL 72 HOUR)
  GROUP BY a.station_name, a.timestamp_utc
)
SELECT
  a.*,
  f.target_low_water_72h
FROM base a
LEFT JOIN future_labels f
  ON a.station_name = f.station_name
 AND a.timestamp_utc = f.timestamp_utc;