import argparse
from datetime import date, timedelta

from ingestion.historical.pegelonline_backfill import run_pegelonline_historical_backfill
from ingestion.sources.dwd import run_dwd_ingestion
from ingestion.sources.pegelonline import run_pegelonline_ingestion
from ingestion.stages.run_sql_stage_1_foundation import run_stage1_sql
from ingestion.stages.run_sql_stage_2_modeling import run_stage2_sql


def default_recent_cutoff() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def run_pegelonline_all(
    hours: int,
    from_date: str,
    to_date: str,
    chunk_months: int,
) -> dict:
    run_pegelonline_historical_backfill(
        from_date=from_date,
        to_date=to_date,
        chunk_months=chunk_months,
    )
    recent_result = run_pegelonline_ingestion(mode="incremental", hours=hours)
    return {
        "source": "pegelonline",
        "mode": "both",
        "historical": {
            "from_date": from_date,
            "to_date": to_date,
            "chunk_months": chunk_months,
        },
        "recent": recent_result or {},
    }


def run_dwd_all() -> dict:
    historical_result = run_dwd_ingestion(mode="historical")
    recent_result = run_dwd_ingestion(mode="recent")
    return {
        "source": "dwd",
        "mode": "both",
        "historical": historical_result or {},
        "recent": recent_result or {},
    }


def run_source_ingestion(
    source: str,
    mode: str,
    hours: int,
    from_date: str | None,
    to_date: str | None,
    chunk_months: int,
) -> dict:
    def as_int(value) -> int:
        if value is None:
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def combine_recent_results(
        pegel_result: dict,
        dwd_result: dict,
    ) -> dict:
        return {
            "source": "all",
            "mode": "recent",
            "rows_ingested": (
                as_int(pegel_result.get("rows_ingested"))
                + as_int(dwd_result.get("rows_ingested"))
            ),
            "stations_processed": max(
                as_int(
                    pegel_result.get("stations_processed")
                ),
                as_int(
                    dwd_result.get("stations_processed")
                ),
            ),
            "stations_failed": (
                as_int(pegel_result.get("stations_failed"))
                + as_int(dwd_result.get("stations_failed"))
            ),
            "pegelonline_rows_ingested": as_int(
                pegel_result.get("rows_ingested")
            ),
            "dwd_rows_ingested": as_int(
                dwd_result.get("rows_ingested")
            ),
            "dwd_proxy_backfilled_rows": as_int(
                dwd_result.get("proxy_backfilled_rows")
            ),
            "pegelonline_watermark_after": (
                pegel_result.get("watermark_after")
            ),
            "dwd_watermark_after": (
                dwd_result.get("watermark_after")
            ),
        }

    if source == "pegelonline":
        if mode == "historical":
            if not from_date or not to_date:
                raise ValueError(
                    "historical mode requires "
                    "--from-date and --to-date"
                )

            run_pegelonline_historical_backfill(
                from_date=from_date,
                to_date=to_date,
                chunk_months=chunk_months,
            )

            return {
                "source": "pegelonline",
                "mode": "historical",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
                "from_date": from_date,
                "to_date": to_date,
                "chunk_months": chunk_months,
            }

        if mode == "recent":
            result = run_pegelonline_ingestion(
                mode="incremental",
                hours=hours,
            )

            return result or {
                "source": "pegelonline",
                "mode": "recent",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
            }

        if mode == "both":
            if not from_date or not to_date:
                raise ValueError(
                    "both mode requires "
                    "--from-date and --to-date"
                )

            result = run_pegelonline_all(
                hours=hours,
                from_date=from_date,
                to_date=to_date,
                chunk_months=chunk_months,
            )

            return result or {
                "source": "pegelonline",
                "mode": "both",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
            }

    if source == "dwd":
        if mode == "historical":
            result = run_dwd_ingestion(
                mode="historical"
            )

            return result or {
                "source": "dwd",
                "mode": "historical",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
            }

        if mode == "recent":
            result = run_dwd_ingestion(
                mode="recent"
            )

            return result or {
                "source": "dwd",
                "mode": "recent",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
            }

        if mode == "both":
            result = run_dwd_all()

            return result or {
                "source": "dwd",
                "mode": "both",
                "rows_ingested": 0,
                "stations_processed": 0,
                "stations_failed": 0,
            }

    if source == "all":
        if mode == "historical":
            if not from_date or not to_date:
                raise ValueError(
                    "historical mode requires "
                    "--from-date and --to-date"
                )

            run_pegelonline_historical_backfill(
                from_date=from_date,
                to_date=to_date,
                chunk_months=chunk_months,
            )

            dwd_result = run_dwd_ingestion(
                mode="historical"
            ) or {}

            return {
                "source": "all",
                "mode": "historical",
                "rows_ingested": as_int(
                    dwd_result.get("rows_ingested")
                ),
                "stations_processed": as_int(
                    dwd_result.get("stations_processed")
                ),
                "stations_failed": as_int(
                    dwd_result.get("stations_failed")
                ),
                "dwd_rows_ingested": as_int(
                    dwd_result.get("rows_ingested")
                ),
                "dwd_proxy_backfilled_rows": as_int(
                    dwd_result.get(
                        "proxy_backfilled_rows"
                    )
                ),
            }

        if mode == "recent":
            pegel_result = (
                run_pegelonline_ingestion(
                    mode="incremental",
                    hours=hours,
                )
                or {}
            )

            dwd_result = (
                run_dwd_ingestion(
                    mode="recent"
                )
                or {}
            )

            return combine_recent_results(
                pegel_result=pegel_result,
                dwd_result=dwd_result,
            )

        if mode == "both":
            if not from_date or not to_date:
                raise ValueError(
                    "both mode requires "
                    "--from-date and --to-date"
                )

            pegel_result = (
                run_pegelonline_all(
                    hours=hours,
                    from_date=from_date,
                    to_date=to_date,
                    chunk_months=chunk_months,
                )
                or {}
            )

            dwd_result = (
                run_dwd_all()
                or {}
            )

            pegel_recent = (
                pegel_result.get("recent")
                or pegel_result
            )

            dwd_recent = (
                dwd_result.get("recent")
                or dwd_result
            )

            return {
                "source": "all",
                "mode": "both",
                "rows_ingested": (
                    as_int(
                        pegel_recent.get(
                            "rows_ingested"
                        )
                    )
                    + as_int(
                        dwd_recent.get(
                            "rows_ingested"
                        )
                    )
                ),
                "stations_processed": max(
                    as_int(
                        pegel_recent.get(
                            "stations_processed"
                        )
                    ),
                    as_int(
                        dwd_recent.get(
                            "stations_processed"
                        )
                    ),
                ),
                "stations_failed": (
                    as_int(
                        pegel_recent.get(
                            "stations_failed"
                        )
                    )
                    + as_int(
                        dwd_recent.get(
                            "stations_failed"
                        )
                    )
                ),
                "pegelonline_rows_ingested": as_int(
                    pegel_recent.get(
                        "rows_ingested"
                    )
                ),
                "dwd_rows_ingested": as_int(
                    dwd_recent.get(
                        "rows_ingested"
                    )
                ),
                "dwd_proxy_backfilled_rows": as_int(
                    dwd_recent.get(
                        "proxy_backfilled_rows"
                    )
                ),
                "pegelonline_watermark_after": (
                    pegel_recent.get(
                        "watermark_after"
                    )
                ),
            }

    raise ValueError(
        "Unsupported source/mode combination: "
        f"source={source}, mode={mode}"
    )

def run_full_pipeline(
    pegel_from_date: str,
    pegel_to_date: str,
    chunk_months: int,
    pegel_hours: int,
    include_dwd: bool,
) -> dict:
    pegel_result = run_pegelonline_all(
        hours=pegel_hours,
        from_date=pegel_from_date,
        to_date=pegel_to_date,
        chunk_months=chunk_months,
    )

    dwd_result = None
    if include_dwd:
        dwd_result = run_dwd_all()

    stage1_result = run_stage1_sql()
    stage2_result = run_stage2_sql()

    return {
        "pegelonline": pegel_result,
        "dwd": dwd_result,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="full",
        choices=["ingestion", "stage1", "stage2", "full"],
        help="Which part of the pipeline to run",
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=["pegelonline", "dwd", "all"],
        help="Source to ingest when step includes ingestion",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["historical", "recent", "both"],
        help="Ingestion mode",
    )
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--from-date", dest="from_date", default="2018-01-01")
    parser.add_argument("--to-date", dest="to_date", default=default_recent_cutoff())
    parser.add_argument("--chunk-months", dest="chunk_months", type=int, default=1)
    parser.add_argument(
        "--skip-dwd",
        action="store_true",
        help="Skip DWD ingestion during full pipeline runs",
    )
    args = parser.parse_args()

    if args.step == "ingestion":
        run_source_ingestion(
            source=args.source,
            mode=args.mode,
            hours=args.hours,
            from_date=args.from_date,
            to_date=args.to_date,
            chunk_months=args.chunk_months,
        )
        return

    if args.step == "stage1":
        run_stage1_sql()
        return

    if args.step == "stage2":
        run_stage2_sql()
        return

    if args.step == "full":
        run_full_pipeline(
            pegel_from_date=args.from_date,
            pegel_to_date=args.to_date,
            chunk_months=args.chunk_months,
            pegel_hours=args.hours,
            include_dwd=not args.skip_dwd,
        )
        return


if __name__ == "__main__":
    main()