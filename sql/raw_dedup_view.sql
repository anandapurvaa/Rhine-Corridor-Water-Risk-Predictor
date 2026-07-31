CREATE OR REPLACE VIEW `rhine-corridor-navigator.rhein_raw.v_pegelonline_measurements_dedup` AS
SELECT *
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY source_record_hash
      ORDER BY ingestion_ts_utc DESC
    ) AS rn
  FROM `rhine-corridor-navigator.rhein_raw.pegelonline_measurements`
)
WHERE rn = 1;