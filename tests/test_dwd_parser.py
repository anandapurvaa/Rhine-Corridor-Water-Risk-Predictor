import io
import zipfile

import pandas as pd

from ingestion.sources.dwd_parser import (
    build_canonical_dwd_frame,
    normalize_station_id,
    parse_dwd_timestamp,
    parse_dwd_zip_bytes,
)


def make_test_zip() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w") as zf:
        content = "STATIONS_ID;MESS_DATUM;QN_9;TT_TU;RF_TU\n01048;2026073112;1;22.4;61\n"
        zf.writestr("produkt_temp.txt", content)
    return mem.getvalue()


def test_parse_dwd_zip_bytes():
    df = parse_dwd_zip_bytes(make_test_zip())
    assert "STATIONS_ID" in df.columns
    assert len(df) == 1


def test_normalize_station_id():
    assert normalize_station_id("183") == "00183"


def test_parse_dwd_timestamp():
    s = pd.Series(["2026073112"])
    out = parse_dwd_timestamp(s)
    assert str(out.iloc[0]).startswith("2026-07-31 12:00:00+00:00")


def test_build_canonical_dwd_frame_air_temperature():
    df = pd.DataFrame(
        [{"STATIONS_ID": "01048", "MESS_DATUM": "2026073112", "QN_9": "1", "TT_TU": "22.4", "RF_TU": "61"}]
    )
    out = build_canonical_dwd_frame(
        df=df,
        variable_family="air_temperature",
        source_url="test-url",
        ingestion_ts_utc="2026-07-31T12:30:00Z",
    )
    assert "temperature_c" in out.columns
    assert "relative_humidity_pct" in out.columns