CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.label_low_water_events` AS
SELECT
  * REPLACE(
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', timestamp_utc) AS timestamp_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', ingestion_ts_utc) AS ingestion_ts_utc,
    ROUND(value, 2) AS value,
    ROUND(latitude, 2) AS latitude,
    ROUND(longitude, 2) AS longitude,
    ROUND(lag_1, 2) AS lag_1,
    ROUND(lag_3, 2) AS lag_3,
    ROUND(lag_6, 2) AS lag_6,
    ROUND(diff_1, 2) AS diff_1,
    ROUND(diff_3, 2) AS diff_3,
    ROUND(rolling_mean_3, 2) AS rolling_mean_3,
    ROUND(rolling_std_3, 2) AS rolling_std_3,
    ROUND(rolling_min_6, 2) AS rolling_min_6,
    ROUND(rolling_max_6, 2) AS rolling_max_6
  ),
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