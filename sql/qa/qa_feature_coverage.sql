SELECT
  COUNT(*) AS total_rows,
  COUNTIF(dwd_station_id IS NOT NULL) AS mapped_rows,
  COUNTIF(primary_dwd_station_id IS NOT NULL) AS primary_mapped_rows,
  COUNTIF(temperature_c IS NOT NULL) AS temp_rows,
  COUNTIF(precipitation_mm IS NOT NULL) AS precip_rows,
  COUNTIF(wind_speed_ms IS NOT NULL) AS wind_rows,
  COUNTIF(pressure_hpa IS NOT NULL) AS pressure_rows,
  COUNTIF(relative_humidity_pct IS NOT NULL) AS humidity_rows,
  COUNTIF(temperature_c_blend IS NOT NULL) AS temp_blend_rows,
  COUNTIF(precipitation_mm_blend IS NOT NULL) AS precip_blend_rows,
  COUNTIF(temp_roll_mean_12 IS NOT NULL) AS temp_roll_12_rows,
  COUNTIF(precip_roll_sum_24 IS NOT NULL) AS precip_roll_24_rows,
  COUNTIF(pressure_delta_1 IS NOT NULL) AS pressure_delta_rows
FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_enriched`;