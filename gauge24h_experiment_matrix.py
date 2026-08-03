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
    "GAUGE24H_MAX_BACKTEST_MONTHS": "18",
}


EXPERIMENTS = [
    {
        "name": "global_default_24h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_default_24h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_delta_24h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_delta_24h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_simple_features_24h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "cluster_simple_features_24h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "global_delta_simple_24h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "cluster_delta_simple_24h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "24",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "global_default_48h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_default_48h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "level",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_delta_48h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "cluster_delta_48h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "default",
        },
    },
    {
        "name": "global_delta_simple_48h",
        "module": "modeling.backtest_walkforward_gauge_24h",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
    {
        "name": "cluster_delta_simple_48h",
        "module": "modeling.backtest_walkforward_gauge_24h_cluster",
        "env": {
            **BASE_ENV,
            "GAUGE24H_HORIZON_HOURS": "48",
            "GAUGE24H_TARGET_MODE": "delta",
            "GAUGE24H_FEATURE_SET": "simple",
        },
    },
]


def parse_last_json_block(stdout: str) -> dict:
    lines = stdout.strip().splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("{"):
            start = i
            break
    if start is None:
        raise ValueError("No JSON block found in stdout")
    blob = "\n".join(lines[start:])
    return json.loads(blob)


def run_experiment(exp: dict) -> dict:
    env = os.environ.copy()
    env.update(exp["env"])
    cmd = [sys.executable, "-m", exp["module"]]
    started_at = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    result = {
        "name": exp["name"],
        "module": exp["module"],
        "env": exp["env"],
        "returncode": proc.returncode,
        "started_at": started_at,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }
    if proc.returncode == 0:
        metrics = parse_last_json_block(proc.stdout)
        result.update(metrics)
    return result


def main() -> int:
    all_results = []
    for exp in EXPERIMENTS:
        print(f"=== {exp['name']} ===", flush=True)
        result = run_experiment(exp)
        all_results.append(result)
        if result["returncode"] == 0:
            key_metrics = {
                k: result.get(k)
                for k in [
                    "mean_model_rmse",
                    "mean_cluster_rmse",
                    "mean_global_rmse",
                    "mean_persist_rmse",
                    "mean_roll_rmse",
                    "model_vs_persist_rmse_gain",
                    "cluster_vs_global_rmse_gain",
                    "cluster_vs_persist_rmse_gain",
                    "mean_cluster_models_used",
                ]
                if k in result
            }
            print(json.dumps(key_metrics, indent=2), flush=True)
        else:
            print(result["stderr_tail"], file=sys.stderr, flush=True)

    out_path = ROOT / "gauge24h_experiment_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0 if all(r["returncode"] == 0 for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())