import pandas as pd

from ingestion.common.bigquery_writer import normalize_pegelonline_dataframe


def test_normalize_pegelonline_dataframe_converts_timestamps():
    df = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-07-31T13:00:00Z",
                "ingestion_ts_utc": "2026-07-31T13:05:00Z",
                "value": "123.4",
                "latitude": "50.0",
                "longitude": "7.6",
                "station_id": "abc",
                "station_name": "KAUB",
                "timeseries_name": "W",
                "unit": "cm",
                "source": "pegelonline",
                "source_record_hash": "hash",
                "source_url": "url",
            }
        ]
    )

    out = normalize_pegelonline_dataframe(df)

    assert str(out["timestamp_utc"].dtype).startswith("datetime64")
    assert str(out["ingestion_ts_utc"].dtype).startswith("datetime64")
    assert str(out["value"].dtype) in ("float64", "Float64")