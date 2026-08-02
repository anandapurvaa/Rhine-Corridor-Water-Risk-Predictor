from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

STEPS = {
    "backtest": "modeling.backtest_walkforward_gauge_24h",
    "train": "modeling.train_gauge_24h_production",
    "predict": "modeling.predict_gauge_24h_production",
    "evaluate": "modeling.evaluate_predictions_gauge_24h",
}

PIPELINES = {
    "full": ["backtest", "train", "predict", "evaluate"],
    "deploy": ["train", "predict"],
    "monitor": ["evaluate"],
    "research": ["backtest"],
}


def run_step(step_name: str) -> None:
    module_name = STEPS[step_name]

    print(f"\n=== Running step: {step_name} ===")
    print(f"Module: {module_name}")

    started = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    elapsed = time.time() - started

    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed with exit code {result.returncode}")

    print(f"=== Completed step: {step_name} in {elapsed:.1f}s ===")


def main():
    parser = argparse.ArgumentParser(description="Run the Gauge 24h production pipeline.")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES.keys()),
        help="Named pipeline to run.",
    )
    parser.add_argument(
        "--step",
        choices=sorted(STEPS.keys()),
        help="Run a single step only.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="When using --pipeline full, skip the backtest stage.",
    )
    args = parser.parse_args()

    if not args.pipeline and not args.step:
        parser.error("Provide either --pipeline or --step")

    if args.pipeline and args.step:
        parser.error("Use either --pipeline or --step, not both")

    if args.step:
        plan = [args.step]
    else:
        plan = PIPELINES[args.pipeline].copy()
        if args.pipeline == "full" and args.skip_backtest:
            plan = [x for x in plan if x != "backtest"]

    print("Planned steps:", " -> ".join(plan))

    for step_name in plan:
        run_step(step_name)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()