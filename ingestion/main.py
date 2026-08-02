import argparse
from datetime import date, timedelta

from ingestion.historical.pegelonline_backfill import run_pegelonline_historical_backfill
from ingestion.sources.dwd import run_dwd_ingestion
from ingestion.sources.pegelonline import run_pegelonline_ingestion
from ingestion.stages.run_sql_stage_1_foundation import run_stage1_sql
from ingestion.stages.run_sql_stage_2_modeling import run_stage2_sql


def default_recent_cutoff() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def run_source_ingestion(
    source: str,
    mode: str,
    hours: int,
    from_date: str | None,
    to_date: str | None,
    chunk_months: int,
) -> None:
    if source == "pegelonline" and mode == "historical":
        if not from_date or not to_date:
            raise ValueError("historical mode requires --from-date and --to-date")
        run_pegelonline_historical_backfill(
            from_date=from_date,
            to_date=to_date,
            chunk_months=chunk_months,
        )
        return

    if source == "pegelonline":
        run_pegelonline_ingestion(mode=mode, hours=hours)
        return

    if source == "dwd":
        run_dwd_ingestion(mode="recent")
        return

    raise ValueError(f"Unsupported source: {source}")


def run_full_pipeline(
    pegel_from_date: str,
    pegel_to_date: str,
    chunk_months: int,
    pegel_hours: int,
    include_dwd: bool,
) -> None:
    run_pegelonline_historical_backfill(
        from_date=pegel_from_date,
        to_date=pegel_to_date,
        chunk_months=chunk_months,
    )
    run_pegelonline_ingestion(mode="incremental", hours=pegel_hours)

    if include_dwd:
        run_dwd_ingestion(mode="recent")

    run_stage1_sql()
    run_stage2_sql()


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
        default="pegelonline",
        choices=["pegelonline", "dwd"],
        help="Source to ingest when step includes ingestion",
    )
    parser.add_argument(
        "--mode",
        default="incremental",
        choices=["incremental", "historical", "recent"],
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