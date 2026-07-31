from ingestion.common.hashing import stable_record_hash


def test_stable_record_hash_same_input_same_hash():
    record = {
        "station_id": "abc",
        "timeseries_name": "W",
        "timestamp_utc": "2026-07-31T10:00:00Z",
        "value": 123.4,
        "source": "pegelonline",
    }

    h1 = stable_record_hash(record, ["station_id", "timeseries_name", "timestamp_utc", "value", "source"])
    h2 = stable_record_hash(record, ["station_id", "timeseries_name", "timestamp_utc", "value", "source"])

    assert h1 == h2


def test_stable_record_hash_changed_value_changes_hash():
    r1 = {
        "station_id": "abc",
        "timeseries_name": "W",
        "timestamp_utc": "2026-07-31T10:00:00Z",
        "value": 123.4,
        "source": "pegelonline",
    }
    r2 = {
        "station_id": "abc",
        "timeseries_name": "W",
        "timestamp_utc": "2026-07-31T10:00:00Z",
        "value": 124.4,
        "source": "pegelonline",
    }

    h1 = stable_record_hash(r1, ["station_id", "timeseries_name", "timestamp_utc", "value", "source"])
    h2 = stable_record_hash(r2, ["station_id", "timeseries_name", "timestamp_utc", "value", "source"])

    assert h1 != h2