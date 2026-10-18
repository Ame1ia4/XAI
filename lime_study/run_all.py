"""Run every model's controlled LIME benchmark, then draw the figures.

Usage (from the repository root):
    python -m lime_study.run_all                  # full study, all models
    python -m lime_study.run_all --quick          # smoke test
    python -m lime_study.run_all --models RandomForest
    python -m lime_study.run_all --plots-only

Each model runs in its own fresh subprocess, sequentially, so timings are not
affected by another model's memory, threads, or warm caches.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from time import perf_counter

from lime_study import config, plot_results
from lime_study.threads import THREAD_ENV_VARS

MODEL_MODULES = {
    "XGBoost": "lime_study.models.run_xgboost",
    "LogisticRegression": "lime_study.models.run_logistic_regression",
    "RandomForest": "lime_study.models.run_random_forest",
}


def run_models(models: list[str], quick: bool) -> None:
    env = {**os.environ, **{name: "1" for name in THREAD_ENV_VARS}}
    for model in models:
        command = [sys.executable, "-m", MODEL_MODULES[model]] + (["--quick"] if quick else [])
        started = perf_counter()
        print(f"=== {model}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env, cwd=config.PACKAGE_DIR.parent)
        print(f"=== {model} finished in {perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled LIME study")
    parser.add_argument(
        "--models", nargs="+", choices=config.MODEL_NAMES, default=list(config.MODEL_NAMES)
    )
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test sweep")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    if not args.plots_only:
        run_models(args.models, args.quick)
    if not args.skip_plots:
        plot_results.main(quick=args.quick)


if __name__ == "__main__":
    main()
