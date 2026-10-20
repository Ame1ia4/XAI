# Rebuild plan: controlled LIME cost/quality study (`lime_study/`)

> **Update 20-10-2026:** the dataset changed from Covertype (54 features) to **UCI HAR (561 features)**. This document reflects the HAR setup. For the reasoning and sources, see [`dataset_justification.md`](dataset_justification.md) and `log_entries/20-10-2026.md`.

## Why we are rebuilding

Paper: *Explainable AI: LIME Cost/Quality Scaling on High-Dimensional Tabular Data across multiple models.*

The paper measures three things:

- **Runtime:** wall-clock time per explanation.
- **Fidelity:** LIME's local surrogate fit, `explanation.score`.
- **Stability:** top-5 Jaccard agreement across repeated runs.

Each is measured across perturbed-feature counts of 100, 300 and 500 and LIME sample budgets from 500 to 10,000. The models are XGBoost, Logistic Regression and Random Forest. TabFM is left out for now.

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
| Data | UCI HAR (DOI 10.24432/C54S4K), downloaded from UCI. Official **subject-based** split: 21 volunteers / 7,352 rows for training, 9 other volunteers / 2,947 rows for testing (≈70/30). No volunteer appears in both. Models train on the full training set. |
| LIME background | 2,000 training rows, seed 42. We do not use the 50,000 from the brief, because the training split only has 7,352 rows. |
| Explained cases | 3 test rows per true class (18 cases), seed 42, chosen without looking at any model's predictions. The predicted class is the label that gets explained. |
| Feature-count axis | Nested subsets of 100, 300 and 500 features: the first k of one seeded permutation of all 561 features. Features outside the subset are held at the case's own values. Models always predict on full 561-column rows, so the models are **not** retrained per feature count. 500 is TabFM's `max_num_features`. |
| Budgets / repeats | `num_samples` of 500, 1,000, 2,500, 5,000 and 10,000. 5 repeats with explainer seeds 42–46. 10 reported features. Top-k for stability is 5. |
| LIME settings (now explicit) | `lime==0.2.0.1`, `discretize_continuous=True`, `discretizer="quartile"`, `kernel_width=None` (LIME default `0.75·sqrt(d)`, where d = number of perturbed features), `feature_selection="auto"` (which resolves to `highest_weights` at 10 features), `sample_around_instance=False`, euclidean distance, Ridge surrogate with `alpha=1`. |
| Runtime fairness | **All models run single-threaded:** `n_jobs=1`, and the BLAS/OpenMP thread environment variables are set to 1. Models run one after another, each in its own process. |
| Model seeds | `random_state=42` on **every** model, including Logistic Regression. |
| Runtime definition | Total wall-clock time of `explain_instance()` (`perf_counter`). This covers perturbation, model probability calls and surrogate fitting. It excludes model training, the warm-up call and explainer construction. We do **not** time individual stages. |
| Fidelity | `explanation.score`, the weighted local R² of the surrogate. Label it "local surrogate fit", not accuracy. We do not measure `local_pred` agreement. |
| Stability | For each case × feature count × budget, all 10 pairs of the 5 repeats are compared using top-5 Jaccard (intersection over union). Spearman was dropped on 20-10-2026 because its values aren't comparable across models (see `log_entries/20-10-2026.md`). |
| Hyperparameters | These are unchanged from the notebooks except for `n_jobs` and `random_state`. Models are **not** tuned or matched to each other. The comparison uses a shared evaluation setup, not equally optimised models. |
| Dataset scope | UCI HAR only (Covertype removed). All dataset-specific values are kept in one block of `config.py`. All features are continuous, so there are no categorical features in the explainer. |

Model definitions (in `config.py`):

- **XGBoost:** `XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)`
- **Logistic Regression:** a `Pipeline` that standardises all 561 columns, followed by `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)`. The model's `n_jobs` setting has no effect with lbfgs, so the thread environment variables are what keep it single-threaded.
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
| `<Model>_raw.csv` | One row per explanation (1,350 rows): case, true/predicted class, feature count, budget, repeat, runtime, fidelity, and the explanation's feature weights as JSON |
| `<Model>_stability.csv` | One row per repeat pair (2,700 rows): top-5 Jaccard |
| `<Model>_summary.csv` | One row per feature count × budget (15 rows): mean/median/std runtime, mean/median/std fidelity, mean/std Jaccard, NaN count |
| `<Model>_performance.json` | Test-set accuracy, balanced accuracy, and macro precision/recall/F1 (2,947 test rows) |
| `<Model>_run_meta.json` | Timestamp, machine/CPU, Python and package versions, data source/DOI/archive SHA-256 and train/test subject IDs, thread settings, config snapshot, model parameters, fit time, total benchmark time |

### Figures (`figures/`)

- `lime_scaling_grid.png/.pdf`: a 3×3 grid. Rows are runtime, fidelity and top-5 Jaccard; columns are 100, 300 and 500 features. The x axis is `num_samples` on a linear scale from 0, and all axes are linear. There is one line per model, with ±1 std bands.
- `cost_vs_quality.png/.pdf`: mean runtime on the x axis (linear) against Jaccard on the y axis, one line per model and feature count, with points ordered by budget. This shows where extra compute stops improving quality.
- `combined_summary.csv` (45 rows) and `performance_table.csv`.

## How to run

From the repository root, with the project venv active:

```bash
pip install -r lime_study/requirements-pinned.txt
python -m lime_study.run_all --quick     # smoke test (~1.5 min); writes to results/quick and figures/quick
python -m lime_study.run_all             # full study: 4,050 explanations
python -m lime_study.run_all --models RandomForest     # one model only
python -m lime_study.run_all --plots-only              # re-plot from saved CSVs
```

Close other heavy programs while the full run is going, because runtime is one of the measurements. Record the machine used: it is written to `run_meta.json` automatically.

## Checks

Each model run asserts the expected counts before it saves anything:

- 1,350 raw rows, 2,700 stability rows and 15 summary rows;
- 18 cases and 90 runs in every group.

`load_har()` also asserts the train/test shapes, the class names, and that no volunteer is in both sets.

It warns if any stability values are NaN. `plot_results.py` checks that the combined summary has 45 rows.

Sanity checks to do by eye after a full run:

- runtime grows roughly linearly with `num_samples`;
- Jaccard rises (or levels off) as the budget increases;
- test accuracy is around 0.9 or higher for every model (the original paper's SVM reached 96%);
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

- The LIME background is 2,000 of the 7,352 training rows.
- The 18 explained cases are balanced by class, unlike the test set, so LIME diagnostic means are averaged over this balanced set.
- Runtime is total explanation time, not split by stage. It includes the cost of building the DataFrame in the shared prediction wrapper, which is the same for every model.
- LIME samples each perturbed feature independently, so perturbations can break the strong correlations between HAR's features (many are statistics of the same signal) and produce unrealistic rows.
- Perturbed-feature subsets are chosen at random (seeded), not by importance, so a 100-feature subset may contain few of the model's most important features.
- Kernel width scales with the number of perturbed features (LIME default). This is part of what the feature-count axis measures.
