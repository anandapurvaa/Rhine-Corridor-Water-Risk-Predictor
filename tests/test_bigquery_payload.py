from ingestion.sources.pegelonline import attach_record_hash


def test_attach_record_hash_adds_hash():
    row = {
        "station_id": "abc",
        "station_name": "KAUB",
        "timeseries_name": "W",
        "timestamp_utc": "2026-07-31T10:00:00Z",
        "value": 130.0,
        "unit": "cm",
        "latitude": 50.0,
        "longitude": 7.0,
        "ingestion_ts_utc": "2026-07-31T10:05:00Z",
        "source": "pegelonline",
        "source_url": "test-url",
    }

    out = attach_record_hash(row)
    assert "source_record_hash" in out
    assert out["source_record_hash"] is not None