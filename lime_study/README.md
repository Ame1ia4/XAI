# lime_study: controlled LIME cost/quality benchmark

This folder holds the controlled experiment for the paper *LIME Cost/Quality Scaling on High-Dimensional Tabular Data across multiple models*. It measures LIME **runtime**, **local fidelity** (`explanation.score`) and **stability** (top-5 Jaccard across repeated runs) for XGBoost, Logistic Regression and Random Forest on **UCI HAR** (561 features, 6 classes, official subject-based train/test split). The sweep covers 100, 300 and 500 perturbed features and 500 to 10,000 LIME samples. The dataset choice and its sources are in [`notes/dataset_justification.md`](notes/dataset_justification.md).

The full methodology, the decisions behind it and the list of changes from the old notebooks are in [`notes/lime_study_rebuild_plan.md`](notes/lime_study_rebuild_plan.md). Only results from this folder should go into the paper. The notebooks in `experiments/` are kept for reference only.

## Run

From the repository root:

```bash
pip install -r lime_study/requirements-pinned.txt
python -m lime_study.run_all --quick      # smoke test, ~1.5 min → results/quick, figures/quick
python -m lime_study.run_all              # full study (1,350 explanations per model)
python -m lime_study.run_all --models RandomForest   # a subset of models
python -m lime_study.run_all --plots-only            # redraw figures from saved results
python -m lime_study.models.run_xgboost              # a single model, without plotting
```

The first run downloads the UCI HAR archive (~60 MB) into `lime_study/data/`, which is gitignored. Close other heavy programs during a full run, because runtime is one of the measurements.

## Files

| File | Role |
|---|---|
| `config.py` | **Every shared value:** seeds, dataset source, LIME background size, feature counts, budgets, repeats, explicit LIME settings, and the three model factories. Edit settings here and nowhere else. |
| `protocol.py` | UCI HAR download/loading (with shape and subject-overlap checks), case selection, nested feature subsets, the LIME benchmark loop, stability metrics, summaries, validation, and saving. Shared by every model. |
| `models/run_*.py` | Thin entry points. Each one only names its model. |
| `run_all.py` | Runs each model in its own subprocess, one after another, then calls `plot_results`. |
| `plot_results.py` | Builds the figures and combined tables from `results/`. |
| `threads.py`, `__init__.py` | Pin BLAS/OpenMP to one thread before numpy loads, so runtimes are comparable across models. |

## Outputs

- `results/<Model>_raw.csv`: one row per explanation, with runtime, fidelity, seed and feature weights.
- `results/<Model>_stability.csv`: one row per pair of repeats, with top-5 Jaccard.
- `results/<Model>_summary.csv`: one row per feature count × budget.
- `results/<Model>_performance.json`: held-out accuracy, balanced accuracy and macro P/R/F1.
- `results/<Model>_run_meta.json`: machine, package versions, thread settings, full config, model parameters and timings.
- `figures/lime_scaling_grid.{png,pdf}`: 3×3 grid of runtime, fidelity and Jaccard against budget, with one column per feature count. All axes are linear.
- `figures/cost_vs_quality.{png,pdf}`: Jaccard against runtime.
- `figures/combined_summary.csv`, `figures/performance_table.csv`.

Each model run checks the expected row counts before saving and fails if they don't match.

## Adding a model or dataset

- **Model:** add a `build_<name>()` factory and a `MODEL_BUILDERS` entry in `config.py`, a style entry in `plot_results.MODEL_STYLE`, a thin `models/run_<name>.py`, and a line in `run_all.MODEL_MODULES`.
- **Dataset:** all dataset-specific values are in the "Dataset block" of `config.py`, with the loader in `protocol.load_har`. Covertype was dropped on 20-10-2026; its old results are in git history (commit `58c9cb4`).

## Troubleshooting

If plotting fails with `Failed to find font DejaVu Sans`, your matplotlib font cache is stale. Delete `~/.matplotlib/fontlist-*.json` and run again.

If the download fails with `CERTIFICATE_VERIFY_FAILED`, install the pinned requirements. The loader uses `certifi`'s certificates, because python.org's macOS Python ships without them.
