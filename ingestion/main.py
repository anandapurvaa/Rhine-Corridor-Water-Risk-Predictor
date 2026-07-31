import argparse
from ingestion.sources.pegelonline import run_pegelonline_ingestion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["pegelonline"])
    parser.add_argument("--mode", default="incremental", choices=["incremental", "backfill"])
    parser.add_argument("--hours", type=int, default=72)
    args = parser.parse_args()

    if args.source == "pegelonline":
        run_pegelonline_ingestion(mode=args.mode, hours=args.hours)


if __name__ == "__main__":
    main()