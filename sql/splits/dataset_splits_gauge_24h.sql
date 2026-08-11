CREATE OR REPLACE TABLE `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h` AS
WITH params AS (
  SELECT 0.70 AS train_fraction, 0.15 AS validation_fraction, 24 * 7 AS gap_hours
),
all_rows AS (
  SELECT *
  FROM `rhine-corridor-navigator.rhein_curated.supervised_gauge_24h_multisource`
  WHERE timestamp_utc IS NOT NULL
),
labeled AS (
  SELECT * FROM all_rows
  WHERE target_value_t_plus_24h IS NOT NULL
),
timeline AS (
  SELECT timestamp_utc,
         ROW_NUMBER() OVER (ORDER BY timestamp_utc) AS rn,
         COUNT(*) OVER () AS n
  FROM (SELECT DISTINCT timestamp_utc FROM labeled)
),
cutoffs AS (
  SELECT
    MIN(CASE WHEN rn >= CAST(n * train_fraction AS INT64) THEN timestamp_utc END) AS train_end_utc,
    MIN(CASE WHEN rn >= CAST(n * (train_fraction + validation_fraction) AS INT64) THEN timestamp_utc END) AS validation_end_utc,
    MIN(gap_hours) AS gap_hours
  FROM timeline CROSS JOIN params
),
boundaries AS (
  SELECT
    train_end_utc,
    TIMESTAMP_ADD(train_end_utc, INTERVAL gap_hours HOUR) AS validation_start_utc,
    validation_end_utc,
    TIMESTAMP_ADD(validation_end_utc, INTERVAL gap_hours HOUR) AS test_start_utc
  FROM cutoffs
),
labeled_split AS (
  SELECT
    l.*,
    CASE
      WHEN l.timestamp_utc < b.train_end_utc THEN 'train'
      WHEN l.timestamp_utc >= b.validation_start_utc
       AND l.timestamp_utc < b.validation_end_utc THEN 'validation'
      WHEN l.timestamp_utc >= b.test_start_utc THEN 'test'
      ELSE 'gap'
    END AS split_name,
    CASE
      WHEN l.timestamp_utc >= b.train_end_utc
       AND l.timestamp_utc < b.validation_start_utc THEN 'train_validation_gap'
      WHEN l.timestamp_utc >= b.validation_end_utc
       AND l.timestamp_utc < b.test_start_utc THEN 'validation_test_gap'
      ELSE NULL
    END AS gap_reason,
    b.train_end_utc,
    b.validation_start_utc,
    b.validation_end_utc,
    b.test_start_utc
  FROM labeled l CROSS JOIN boundaries b
),
production_candidates AS (
  SELECT * EXCEPT(row_number)
  FROM (
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY a.station_name, a.timeseries_name
        ORDER BY a.timestamp_utc DESC
      ) AS row_number
    FROM all_rows a
    CROSS JOIN boundaries b
    WHERE a.timestamp_utc >= b.test_start_utc
      AND a.temperature_c IS NOT NULL
      AND a.precipitation_mm IS NOT NULL
      AND a.wind_speed_ms IS NOT NULL
      AND a.pressure_hpa IS NOT NULL
      AND a.relative_humidity_pct IS NOT NULL
  )
  WHERE row_number = 1
),
production_split AS (
  SELECT
    p.*,
    'production' AS split_name,
    CAST(NULL AS STRING) AS gap_reason,
    b.train_end_utc,
    b.validation_start_utc,
    b.validation_end_utc,
    b.test_start_utc
  FROM production_candidates p CROSS JOIN boundaries b
)
SELECT
  * REPLACE(
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', timestamp_utc) AS timestamp_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', train_end_utc) AS train_end_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', validation_start_utc) AS validation_start_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', validation_end_utc) AS validation_end_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', test_start_utc) AS test_start_utc,
    ROUND(target_value, 2) AS target_value,
    ROUND(latitude, 2) AS latitude,
    ROUND(longitude, 2) AS longitude,
    ROUND(distance_km, 2) AS distance_km,
    ROUND(primary_distance_km, 2) AS primary_distance_km,
    ROUND(backup1_distance_km, 2) AS backup1_distance_km,
    ROUND(backup2_distance_km, 2) AS backup2_distance_km,
    ROUND(temperature_c, 2) AS temperature_c,
    ROUND(precipitation_mm, 2) AS precipitation_mm,
    ROUND(wind_speed_ms, 2) AS wind_speed_ms,
    ROUND(pressure_hpa, 2) AS pressure_hpa,
    ROUND(relative_humidity_pct, 2) AS relative_humidity_pct,
    ROUND(temperature_c_blend, 2) AS temperature_c_blend,
    ROUND(precipitation_mm_blend, 2) AS precipitation_mm_blend,
    ROUND(wind_speed_ms_blend, 2) AS wind_speed_ms_blend,
    ROUND(pressure_hpa_blend, 2) AS pressure_hpa_blend,
    ROUND(relative_humidity_pct_blend, 2) AS relative_humidity_pct_blend,
    ROUND(lag_1, 2) AS lag_1,
    ROUND(lag_3, 2) AS lag_3,
    ROUND(lag_6, 2) AS lag_6,
    ROUND(diff_1, 2) AS diff_1,
    ROUND(diff_3, 2) AS diff_3,
    ROUND(rolling_mean_3, 2) AS rolling_mean_3,
    ROUND(rolling_std_3, 2) AS rolling_std_3,
    ROUND(rolling_min_6, 2) AS rolling_min_6,
    ROUND(rolling_max_6, 2) AS rolling_max_6,
    ROUND(temp_lag_1, 2) AS temp_lag_1,
    ROUND(temp_lag_3, 2) AS temp_lag_3,
    ROUND(temp_lag_6, 2) AS temp_lag_6,
    ROUND(temp_lag_12, 2) AS temp_lag_12,
    ROUND(precip_lag_1, 2) AS precip_lag_1,
    ROUND(precip_lag_3, 2) AS precip_lag_3,
    ROUND(precip_lag_6, 2) AS precip_lag_6,
    ROUND(precip_lag_12, 2) AS precip_lag_12,
    ROUND(wind_lag_1, 2) AS wind_lag_1,
    ROUND(wind_lag_3, 2) AS wind_lag_3,
    ROUND(wind_lag_6, 2) AS wind_lag_6,
    ROUND(pressure_lag_1, 2) AS pressure_lag_1,
    ROUND(pressure_lag_3, 2) AS pressure_lag_3,
    ROUND(humidity_lag_1, 2) AS humidity_lag_1,
    ROUND(temp_roll_mean_3, 2) AS temp_roll_mean_3,
    ROUND(temp_roll_mean_6, 2) AS temp_roll_mean_6,
    ROUND(temp_roll_mean_12, 2) AS temp_roll_mean_12,
    ROUND(precip_roll_mean_6, 2) AS precip_roll_mean_6,
    ROUND(precip_roll_mean_12, 2) AS precip_roll_mean_12,
    ROUND(precip_roll_sum_6, 2) AS precip_roll_sum_6,
    ROUND(precip_roll_sum_12, 2) AS precip_roll_sum_12,
    ROUND(precip_roll_sum_24, 2) AS precip_roll_sum_24,
    ROUND(wind_roll_mean_6, 2) AS wind_roll_mean_6,
    ROUND(wind_roll_mean_12, 2) AS wind_roll_mean_12,
    ROUND(pressure_roll_mean_6, 2) AS pressure_roll_mean_6,
    ROUND(pressure_roll_mean_12, 2) AS pressure_roll_mean_12,
    ROUND(humidity_roll_mean_6, 2) AS humidity_roll_mean_6,
    ROUND(humidity_roll_mean_12, 2) AS humidity_roll_mean_12,
    ROUND(pressure_delta_1, 2) AS pressure_delta_1,
    ROUND(pressure_delta_3, 2) AS pressure_delta_3,
    ROUND(temp_change_1_3, 2) AS temp_change_1_3,
    ROUND(precip_accel_12_24, 2) AS precip_accel_12_24,
    ROUND(target_value_t_plus_24h, 2) AS target_value_t_plus_24h
  )
FROM labeled_split
UNION ALL BY NAME
SELECT
  * REPLACE(
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', timestamp_utc) AS timestamp_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', train_end_utc) AS train_end_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', validation_start_utc) AS validation_start_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', validation_end_utc) AS validation_end_utc,
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', test_start_utc) AS test_start_utc,
    ROUND(target_value, 2) AS target_value,
    ROUND(latitude, 2) AS latitude,
    ROUND(longitude, 2) AS longitude,
    ROUND(distance_km, 2) AS distance_km,
    ROUND(primary_distance_km, 2) AS primary_distance_km,
    ROUND(backup1_distance_km, 2) AS backup1_distance_km,
    ROUND(backup2_distance_km, 2) AS backup2_distance_km,
    ROUND(temperature_c, 2) AS temperature_c,
    ROUND(precipitation_mm, 2) AS precipitation_mm,
    ROUND(wind_speed_ms, 2) AS wind_speed_ms,
    ROUND(pressure_hpa, 2) AS pressure_hpa,
    ROUND(relative_humidity_pct, 2) AS relative_humidity_pct,
    ROUND(temperature_c_blend, 2) AS temperature_c_blend,
    ROUND(precipitation_mm_blend, 2) AS precipitation_mm_blend,
    ROUND(wind_speed_ms_blend, 2) AS wind_speed_ms_blend,
    ROUND(pressure_hpa_blend, 2) AS pressure_hpa_blend,
    ROUND(relative_humidity_pct_blend, 2) AS relative_humidity_pct_blend,
    ROUND(lag_1, 2) AS lag_1,
    ROUND(lag_3, 2) AS lag_3,
    ROUND(lag_6, 2) AS lag_6,
    ROUND(diff_1, 2) AS diff_1,
    ROUND(diff_3, 2) AS diff_3,
    ROUND(rolling_mean_3, 2) AS rolling_mean_3,
    ROUND(rolling_std_3, 2) AS rolling_std_3,
    ROUND(rolling_min_6, 2) AS rolling_min_6,
    ROUND(rolling_max_6, 2) AS rolling_max_6,
    ROUND(temp_lag_1, 2) AS temp_lag_1,
    ROUND(temp_lag_3, 2) AS temp_lag_3,
    ROUND(temp_lag_6, 2) AS temp_lag_6,
    ROUND(temp_lag_12, 2) AS temp_lag_12,
    ROUND(precip_lag_1, 2) AS precip_lag_1,
    ROUND(precip_lag_3, 2) AS precip_lag_3,
    ROUND(precip_lag_6, 2) AS precip_lag_6,
    ROUND(precip_lag_12, 2) AS precip_lag_12,
    ROUND(wind_lag_1, 2) AS wind_lag_1,
    ROUND(wind_lag_3, 2) AS wind_lag_3,
    ROUND(wind_lag_6, 2) AS wind_lag_6,
    ROUND(pressure_lag_1, 2) AS pressure_lag_1,
    ROUND(pressure_lag_3, 2) AS pressure_lag_3,
    ROUND(humidity_lag_1, 2) AS humidity_lag_1,
    ROUND(temp_roll_mean_3, 2) AS temp_roll_mean_3,
    ROUND(temp_roll_mean_6, 2) AS temp_roll_mean_6,
    ROUND(temp_roll_mean_12, 2) AS temp_roll_mean_12,
    ROUND(precip_roll_mean_6, 2) AS precip_roll_mean_6,
    ROUND(precip_roll_mean_12, 2) AS precip_roll_mean_12,
    ROUND(precip_roll_sum_6, 2) AS precip_roll_sum_6,
    ROUND(precip_roll_sum_12, 2) AS precip_roll_sum_12,
    ROUND(precip_roll_sum_24, 2) AS precip_roll_sum_24,
    ROUND(wind_roll_mean_6, 2) AS wind_roll_mean_6,
    ROUND(wind_roll_mean_12, 2) AS wind_roll_mean_12,
    ROUND(pressure_roll_mean_6, 2) AS pressure_roll_mean_6,
    ROUND(pressure_roll_mean_12, 2) AS pressure_roll_mean_12,
    ROUND(humidity_roll_mean_6, 2) AS humidity_roll_mean_6,
    ROUND(humidity_roll_mean_12, 2) AS humidity_roll_mean_12,
    ROUND(pressure_delta_1, 2) AS pressure_delta_1,
    ROUND(pressure_delta_3, 2) AS pressure_delta_3,
    ROUND(temp_change_1_3, 2) AS temp_change_1_3,
    ROUND(precip_accel_12_24, 2) AS precip_accel_12_24,
    ROUND(target_value_t_plus_24h, 2) AS target_value_t_plus_24h
  )
FROM production_split;