from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent

BASE_ENV = {
    "GAUGE24H_STEP_DAYS": "30",
    "GAUGE24H_MAX_BACKTEST_MONTHS": "0",
    "GAUGE24H_BACKTEST_END_UTC": (
        "2025-12-31T23:59:59Z"
    ),
}


EXPERIMENTS = [
    {
        "name": "global_delta_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_delta_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h_cluster"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_level_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_level_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h_cluster"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_delta_simple_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "cluster_delta_simple_24h",
        "module": (
            "modeling.backtest_walkforward_gauge_24h_cluster"
        ),
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
]


def parse_last_json_block(stdout: str) -> dict:
    lines = stdout.strip().splitlines()

    for start in range(len(lines) - 1, -1, -1):
        if lines[start].strip() != "{":
            continue

        blob = "\n".join(lines[start:])

        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    raise ValueError(
        "No valid JSON summary found in stdout"
    )


def run_experiment(exp: dict) -> dict:
    env = os.environ.copy()
    env.update(exp["env"])

    command = [
        sys.executable,
        "-m",
        exp["module"],
    ]

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )

    result = {
        "name": exp["name"],
        "module": exp["module"],
        "env": exp["env"],
        "returncode": process.returncode,
        "started_at_utc": started_at,
        "stdout_tail": "\n".join(
            process.stdout.splitlines()[-40:]
        ),
        "stderr_tail": "\n".join(
            process.stderr.splitlines()[-40:]
        ),
    }

    if process.returncode == 0:
        metrics = parse_last_json_block(
            process.stdout
        )
        result.update(metrics)

    return result


def select_key_metrics(result: dict) -> dict:
    keys = [
        "backtest_end_utc",
        "selection_purpose",
        "horizon_hours",
        "target_mode",
        "feature_set",
        "folds",
        "rows_scored",
        "mean_model_rmse",
        "mean_cluster_rmse",
        "mean_global_rmse",
        "mean_persist_rmse",
        "mean_roll_rmse",
        "model_vs_persist_rmse_gain",
        "model_vs_roll_rmse_gain",
        "cluster_vs_global_rmse_gain",
        "cluster_vs_persist_rmse_gain",
        "mean_cluster_models_used",
    ]

    return {
        key: result.get(key)
        for key in keys
        if key in result
    }


def main() -> int:
    all_results = []

    for experiment in EXPERIMENTS:
        print(
            f"=== {experiment['name']} ===",
            flush=True,
        )

        result = run_experiment(
            experiment
        )
        all_results.append(result)

        if result["returncode"] == 0:
            print(
                json.dumps(
                    select_key_metrics(result),
                    indent=2,
                ),
                flush=True,
            )
        else:
            print(
                result["stderr_tail"],
                file=sys.stderr,
                flush=True,
            )

    output_path = (
        ROOT
        / "gauge24h_experiment_results.json"
    )

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")

    return int(
        not all(
            result["returncode"] == 0
            for result in all_results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())