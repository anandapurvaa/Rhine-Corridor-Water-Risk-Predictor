import pandas as pd

from ingestion.historical.archive_parser import normalize_archive_dataframe, parse_archive_timestamp_to_utc


def test_parse_archive_timestamp_to_utc():
    out = parse_archive_timestamp_to_utc("2026-01-15 12:00", timezone_name="Europe/Berlin")
    assert out.startswith("2026-01-15T11:00:00")


def test_normalize_archive_dataframe():
    df = pd.DataFrame(
        [
            {"timestamp": "2026-01-15 12:00", "value": "123.4"},
            {"timestamp": "2026-01-15 13:00", "value": "124.5"},
        ]
    )

    out = normalize_archive_dataframe(
        df=df,
        station_id="abc",
        station_name="KAUB",
        timeseries_name="W",
        unit="cm",
        source_url="test-url",
        ingestion_ts_utc="2026-07-31T15:00:00Z",
    )

    assert len(out) == 2
    assert "source_record_hash" in out.columns
    assert out["station_name"].iloc[0] == "KAUB"