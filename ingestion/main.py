import argparse

from ingestion.historical.pegelonline_backfill import run_pegelonline_historical_backfill
from ingestion.sources.dwd import run_dwd_ingestion
from ingestion.sources.pegelonline import run_pegelonline_ingestion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["pegelonline", "dwd"])
    parser.add_argument("--mode", default="incremental", choices=["incremental", "backfill", "historical", "recent"])
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--from-date", dest="from_date", default=None)
    parser.add_argument("--to-date", dest="to_date", default=None)
    parser.add_argument("--chunk-months", dest="chunk_months", type=int, default=1)
    args = parser.parse_args()

    if args.source == "pegelonline" and args.mode == "historical":
        if not args.from_date or not args.to_date:
            raise ValueError("historical mode requires --from-date and --to-date")
        run_pegelonline_historical_backfill(
            from_date=args.from_date,
            to_date=args.to_date,
            chunk_months=args.chunk_months,
        )
        return

    if args.source == "pegelonline":
        run_pegelonline_ingestion(mode=args.mode, hours=args.hours)
        return

    if args.source == "dwd":
        run_dwd_ingestion(mode=args.mode)
        return


if __name__ == "__main__":
    main()