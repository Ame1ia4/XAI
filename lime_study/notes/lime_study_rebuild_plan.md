# Rebuild plan: controlled LIME cost/quality study (`lime_study/`)

## Why we are rebuilding

Paper: *Explainable AI: LIME Cost/Quality Scaling on High-Dimensional Tabular Data across multiple models.*

The paper measures three things:

- **Runtime:** wall-clock time per explanation.
- **Fidelity:** LIME's local surrogate fit, `explanation.score`.
- **Stability:** top-5 Jaccard and Spearman agreement across repeated runs.

Each is measured across perturbed-feature counts of 18, 36 and 54 and LIME sample budgets from 500 to 10,000. The models are XGBoost, Logistic Regression and Random Forest. TabFM is left out for now.

The notebooks in `experiments/` are not usable as the controlled study. `methodology_comparison.md` and `experiment_handoff.md` in this folder list the problems:

- Outdated exploratory cells (single-row runs, top-5/5 overlap instead of Jaccard, timing that includes explainer construction, full-data splits) are mixed in with the controlled benchmark call.
- None of the notebooks saved the controlled results.
- LIME settings are left at implicit defaults, and package versions are not pinned or recorded.
- Model randomness and threading are inconsistent: Logistic Regression has no `random_state`, and only XGBoost sets `n_jobs` (to 4).
- The pairwise stability rows are computed and then thrown away.

`lime_study/` is a clean rebuild. The old `experiments/` and `coverTree_activities/` folders are left untouched for reference, but **only results produced by `lime_study/` should go into the paper.**

## Agreed decisions

| Topic | Decision |
|---|---|
| Data | Covertype (`fetch_covtype`). Stratified 10,000-row sample (seed 42), then stratified 80/20 split (8,000 train / 2,000 test, seed 42). The methodology must state that models are trained on this sample, not on all 581,012 rows. |
| LIME background | 2,000 training rows, seed 42. We do not use the 50,000 from the brief, because the training split only has 8,000 rows. |
| Explained cases | 3 test rows per true class (21 cases), seed 42, chosen without looking at any model's predictions. The predicted class is the label that gets explained. |
| Feature-count axis | Nested subsets of 18, 36 and 54 features. All 10 numeric features are always included, plus binary indicators taken from one seeded permutation. Features outside the subset are held at the case's own values. Models always predict on full 54-column rows, so the models are **not** retrained per feature count. |
| Budgets / repeats | `num_samples` of 500, 1,000, 2,500, 5,000 and 10,000. 5 repeats with explainer seeds 42–46. 10 reported features. Top-k for stability is 5. |
| LIME settings (now explicit) | `lime==0.2.0.1`, `discretize_continuous=True`, `discretizer="quartile"`, `kernel_width=None` (LIME default `0.75·sqrt(d)`, where d = number of perturbed features), `feature_selection="auto"` (which resolves to `highest_weights` at 10 features), `sample_around_instance=False`, euclidean distance, Ridge surrogate with `alpha=1`. |
| Runtime fairness | **All models run single-threaded:** `n_jobs=1`, and the BLAS/OpenMP thread environment variables are set to 1. Models run one after another, each in its own process. |
| Model seeds | `random_state=42` on **every** model, including Logistic Regression. |
| Runtime definition | Total wall-clock time of `explain_instance()` (`perf_counter`). This covers perturbation, model probability calls and surrogate fitting. It excludes model training, the warm-up call and explainer construction. We do **not** time individual stages. |
| Fidelity | `explanation.score`, the weighted local R² of the surrogate. Label it "local surrogate fit", not accuracy. We do not measure `local_pred` agreement. |
| Stability | For each case × feature count × budget, all 10 pairs of the 5 repeats are compared using top-5 Jaccard (intersection over union) and Spearman over the union of selected features. |
| Hyperparameters | These are unchanged from the notebooks except for `n_jobs` and `random_state`. Models are **not** tuned or matched to each other. The comparison uses a shared evaluation setup, not equally optimised models. |
| Dataset scope | Covertype only. All dataset-specific values are kept in one block of `config.py`, so a higher-dimensional dataset can be added later. |

Model definitions (in `config.py`):

- **XGBoost:** `XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)`
- **Logistic Regression:** a `Pipeline` that standardises the 10 numeric columns and passes the binary columns through unchanged, followed by `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)`. The model's `n_jobs` setting has no effect with lbfgs, so the thread environment variables are what keep it single-threaded.
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)`

## Folder layout

```
lime_study/
  README.md                  how to run; what each file does
  requirements-pinned.txt    exact package versions used
  __init__.py                forces single-threaded BLAS/OpenMP before numpy loads
  threads.py                 the thread environment-variable helper
  config.py                  ALL shared constants, LIME settings and model factories
  protocol.py                data, case selection, feature subsets, LIME benchmark, metrics, saving, validation
  models/
    run_xgboost.py           thin wrappers: each one only names its model
    run_logistic_regression.py
    run_random_forest.py
  run_all.py                 runs each model in its own subprocess, one after another, then plots
  plot_results.py            reads results/ and writes the figures/
  results/                   per-model output files (see below)
  figures/                   final figures and combined tables
```

**One rule:** a model script contains no data, LIME or sweep logic. Everything shared lives in `config.py` and `protocol.py`, so the three models cannot drift apart.

### Files saved for each model (`results/`)

| File | Contents |
|---|---|
| `<Model>_raw.csv` | One row per explanation (1,575 rows): case, true/predicted class, feature count, budget, repeat, runtime, fidelity, and the explanation's feature weights as JSON |
| `<Model>_stability.csv` | One row per repeat pair (3,150 rows): Spearman and top-5 Jaccard |
| `<Model>_summary.csv` | One row per feature count × budget (15 rows): mean/median/std runtime, mean/median/std fidelity, mean/std Spearman and Jaccard, NaN counts |
| `<Model>_performance.json` | Test-set accuracy, balanced accuracy, and macro precision/recall/F1 (2,000 rows) |
| `<Model>_run_meta.json` | Timestamp, machine/CPU, Python and package versions, thread settings, config snapshot, model parameters, fit time, total benchmark time |

### Figures (`figures/`)

- `lime_scaling_grid.png/.pdf`: a 3×3 grid. Rows are runtime, fidelity and top-5 Jaccard; columns are 18, 36 and 54 features. The x axis is `num_samples` on a log scale. There is one line per model, with ±1 std bands.
- `cost_vs_quality.png/.pdf`: mean runtime on the x axis (log) against Jaccard on the y axis, one line per model and feature count, with points ordered by budget. This shows where extra compute stops improving quality.
- `stability_spearman.png/.pdf`: the Spearman version of the stability row.
- `combined_summary.csv` (45 rows) and `performance_table.csv`.

## How to run

From the repository root, with the project venv active:

```bash
pip install -r lime_study/requirements-pinned.txt
python -m lime_study.run_all --quick     # smoke test (~1 min); writes to results/quick and figures/quick
python -m lime_study.run_all             # full study: 4,725 explanations
python -m lime_study.run_all --models RandomForest     # one model only
python -m lime_study.run_all --plots-only              # re-plot from saved CSVs
```

Close other heavy programs while the full run is going, because runtime is one of the measurements. Record the machine used: it is written to `run_meta.json` automatically.

## Checks

Each model run asserts the expected counts before it saves anything:

- 1,575 raw rows, 3,150 stability rows and 15 summary rows;
- 21 cases and 105 runs in every group.

It warns if any stability values are NaN. `plot_results.py` checks that the combined summary has 45 rows.

Sanity checks to do by eye after a full run:

- runtime grows roughly linearly with `num_samples`;
- Jaccard rises (or levels off) as the budget increases;
- RF and XGBoost accuracy is above Logistic Regression's;
- `run_meta.json` shows `lime 0.2.0.1` and the thread settings at 1;
- re-running one model gives identical fidelity and stability, and only the runtimes change.

## Changes from the old experiments

- LIME constructor and surrogate settings are passed explicitly instead of relying on library defaults.
- `n_jobs=1` and single-threaded BLAS for every model. Previously XGBoost used 4 threads and the other models used library defaults.
- `random_state=42` has been added to Logistic Regression.
- Pairwise stability rows and per-explanation feature weights are now saved, not thrown away.
- Package versions, machine details and the full config are saved with every run.
- One shared `predict_proba` wrapper (numpy rows → named DataFrame) for every model.
- The exploratory cells (single-row demos, top-5/5 overlap, explainer-construction timing, full-data splits) have been dropped.
- Each model runs in a fresh process, one at a time, so models don't affect each other's timings.

## Limitations to state in the paper

- Models are trained on a 10,000-row sample, and the LIME background is 2,000 rows.
- The 21 explained cases are balanced by class, unlike the test set, so LIME diagnostic means are averaged over this balanced set.
- Runtime is total explanation time, not split by stage. It includes the cost of building the DataFrame in the shared prediction wrapper, which is the same for every model.
- LIME perturbations can create combinations of binary wilderness/soil indicators that never occur in the real data.
- Kernel width scales with the number of perturbed features (LIME default). This is part of what the feature-count axis measures.
