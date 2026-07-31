SELECT
  station_name,
  dwd_station_name,
  COUNT(*) AS rows_n,
  COUNTIF(temperature_c IS NOT NULL) AS temp_rows,
  COUNTIF(precipitation_mm IS NOT NULL) AS precip_rows,
  COUNTIF(wind_speed_ms IS NOT NULL) AS wind_rows,
  COUNTIF(pressure_hpa IS NOT NULL) AS pressure_rows,
  COUNTIF(relative_humidity_pct IS NOT NULL) AS humidity_rows
FROM `rhine-corridor-navigator.rhein_curated.feature_gauge_weather_enriched`
GROUP BY station_name, dwd_station_name
ORDER BY rows_n DESC;