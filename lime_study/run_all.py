"""Run every model's controlled LIME benchmark, then draw the figures.

Usage (from the repository root):
    python -m lime_study.run_all                  # full main study, all models
    python -m lime_study.run_all --quick          # smoke test
    python -m lime_study.run_all --models RandomForest
    python -m lime_study.run_all --plots-only
    python -m lime_study.run_all --variants main ranked_subset fixed_kernel

All models run together in one fresh subprocess as a randomized complete block
design: each block (case x feature count x budget x repeat) runs every model once,
blocks run in a seeded random order, and models run in a random order inside each
block. Machine drift therefore affects every model alike. Speed probes through the
run record how much drift there was.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from time import perf_counter

from lime_study import config, plot_results
from lime_study.threads import THREAD_ENV_VARS


def run_models(
    models: list[str], quick: bool, variant: str, allow_version_mismatch: bool = False
) -> None:
    env = {**os.environ, **{name: "1" for name in THREAD_ENV_VARS}}
    command = [sys.executable, "-m", "lime_study.protocol", "--models", *models, "--variant", variant]
    command += ["--quick"] if quick else []
    command += ["--allow-version-mismatch"] if allow_version_mismatch else []
    started = perf_counter()
    print(f"=== {variant}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env, cwd=config.PACKAGE_DIR.parent)
    print(f"=== {variant} finished in {perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled LIME study")
    parser.add_argument(
        "--models", nargs="+", choices=config.MODEL_NAMES, default=list(config.MODEL_NAMES)
    )
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test sweep")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(config.VARIANTS),
        default=[config.MAIN_VARIANT.name],
        help="main study and/or sensitivity checks (ranked_subset, fixed_kernel)",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="run a full sweep outside the pinned environment (not for the paper)",
    )
    args = parser.parse_args()

    # Run every variant before plotting any: plotting checks that all variants on disk
    # explain the same cases, so stale results from another variant would stop the run.
    if not args.plots_only:
        for variant in args.variants:
            run_models(args.models, args.quick, variant, args.allow_version_mismatch)
    if not args.skip_plots:
        for variant in args.variants:
            plot_results.main(quick=args.quick, variant_name=variant)


if __name__ == "__main__":
    main()
