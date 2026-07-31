from ingestion.historical.pegelonline_backfill import month_chunks


def test_month_chunks_single_month():
    chunks = month_chunks("2026-01-01", "2026-01-31", chunk_months=1)
    assert len(chunks) == 1
    assert str(chunks[0][0].date()) == "2026-01-01"
    assert str(chunks[0][1].date()) == "2026-01-31"


def test_month_chunks_two_months():
    chunks = month_chunks("2026-01-01", "2026-03-15", chunk_months=1)
    assert len(chunks) == 3
    assert str(chunks[-1][1].date()) == "2026-03-15"