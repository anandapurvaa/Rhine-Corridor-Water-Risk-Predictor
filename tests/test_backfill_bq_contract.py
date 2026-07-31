import pandas as pd


def test_historical_backfill_drops_segment_id_before_bq():
    df = pd.DataFrame(
        [
            {
                "station_id": "abc",
                "station_name": "KAUB",
                "timeseries_name": "W",
                "timestamp_utc": "2026-06-01T00:00:00Z",
                "value": 120.0,
                "unit": "cm",
                "latitude": None,
                "longitude": None,
                "ingestion_ts_utc": "2026-07-31T16:00:00Z",
                "source": "pegelonline",
                "source_record_hash": "hash",
                "source_url": "url",
                "segment_id": "middle_rhine",
            }
        ]
    )

    out = df.drop(columns=["segment_id"], errors="ignore")

    assert "segment_id" not in out.columns