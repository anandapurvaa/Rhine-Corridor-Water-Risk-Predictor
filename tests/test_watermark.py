from ingestion.common.watermark import get_watermark, set_watermark


def test_watermark_roundtrip():
    payload = {
        "last_successful_ingestion_ts_utc": "2026-07-31T13:00:00Z",
        "rows_written": 43,
    }

    set_watermark("test_source", payload)
    out = get_watermark("test_source")

    assert out == payload