CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.label_low_water_events` AS
SELECT
  *,
  CASE
    WHEN station_name = 'KAUB' AND value <= 120 THEN 1
    WHEN station_name = 'MAXAU' AND value <= 380 THEN 1
    WHEN station_name = 'KOBLENZ' AND value <= 150 THEN 1
    WHEN station_name = 'DUISBURG-RUHRORT' AND value <= 260 THEN 1
    WHEN station_name = 'EMMERICH' AND value <= 140 THEN 1
    WHEN station_name = 'KÖLN' AND value <= 180 THEN 1
    WHEN station_name = 'MAINZ' AND value <= 170 THEN 1
    WHEN station_name = 'WORMS' AND value <= 120 THEN 1
    WHEN station_name = 'SPEYER' AND value <= 200 THEN 1
    WHEN station_name = 'BONN' AND value <= 170 THEN 1
    WHEN station_name = 'DÜSSELDORF' AND value <= 190 THEN 1
    WHEN station_name = 'REES' AND value <= 160 THEN 1
    ELSE 0
  END AS low_water_label
FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_timeseries`
WHERE timeseries_name = 'W';