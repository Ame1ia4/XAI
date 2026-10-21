# Rebuild plan: controlled LIME cost/quality study (`lime_study/`)

> **Update 21-10-2026 (`claim-safe-fixes` branch):** the design was revised so the study's claims hold up. Quality is now scored on held-out samples and against large-N references; all models explain the same class and run interleaved; uncertainty comes from a case-level bootstrap with Holm-corrected tests; the sweep has 5 cases per class and d = 50 added. Every fix and its sources are in [`claim_safe_fixes.md`](claim_safe_fixes.md); every design parameter and its rationale is in [`design_decisions.md`](design_decisions.md). This document reflects the revised design.
>
> **Update 20-10-2026:** the dataset changed from Covertype (54 features) to **UCI HAR (561 features)**. See [`dataset_justification.md`](dataset_justification.md) and `log_entries/20-10-2026.md`.

## Why we are rebuilding

Paper: *Explainable AI: LIME Cost/Quality Scaling on High-Dimensional Tabular Data across multiple models.*

The study answers three questions:

1. How does runtime per explanation scale with N (LIME samples) and d (perturbed features)?
2. How does explanation quality change with N and d?
3. How does the black-box model change the cost/quality trade-off?

It measures:

- **Runtime:** wall-clock time per explanation, and scaling exponents in N and d.
- **Fidelity and error:** the surrogate's kernel-weighted R² and RMSE on held-out perturbations.
- **Convergence:** top-5 Jaccard against two 50,000-sample reference explanations, checked against how well the references agree with each other.
- **Stability:** top-5 Jaccard between repeated runs.

These are measured across perturbed-feature counts of 50, 100, 300 and 500 and LIME sample budgets from 500 to 10,000. The models are XGBoost, Logistic Regression and Random Forest. TabFM is left out for now.

The notebooks in `experiments/` are not usable as the controlled study. `methodology_comparison.md` and `experiment_handoff.md` in this folder (both superseded) list the problems:

- Outdated exploratory cells (single-row runs, top-5/5 overlap instead of Jaccard, timing that includes explainer construction, full-data splits) are mixed in with the controlled benchmark call.
- None of the notebooks saved the controlled results.
- LIME settings are left at implicit defaults, and package versions are not pinned or recorded.
- Model randomness and threading are inconsistent: Logistic Regression has no `random_state`, and only XGBoost sets `n_jobs` (to 4).
- The pairwise stability rows are computed and then thrown away.

`lime_study/` is a clean rebuild. The old `experiments/` and `coverTree_activities/` folders are kept for reference only. **Only results produced by `lime_study/` with the current code should go into the paper.** Results saved before 21-10-2026 use the old design.

## Agreed decisions

The full rationale and sources for each are in [`design_decisions.md`](design_decisions.md).

| Topic | Decision |
|---|---|
| Data | UCI HAR (DOI 10.24432/C54S4K), downloaded from UCI. Official **subject-based** split: 21 volunteers / 7,352 rows for training, 9 other volunteers / 2,947 rows for testing (≈70/30). No volunteer appears in both. Models train on the full training set. |
| LIME background | 2,000 training rows, seed 42. We do not use the 50,000 from the brief, because the training split only has 7,352 rows. |
| Explained cases | **5** test rows per true class (**30** cases), seeded, chosen before any model is fitted. **Every model explains the case's true class**, so all models explain the same quantity; whether each model predicted it is recorded. |
| Feature-count axis | Nested subsets of **50**, 100, 300 and 500 features: prefixes of one seeded permutation of all 561 features. Features outside the subset are held at the case's own values. Models always predict on full 561-column rows and are **not** retrained per feature count. 500 is TabFM's `max_num_features`; 50 adds a fourth point for the runtime exponent in d. |
| Budgets / repeats | `num_samples` of 500, 1,000, 2,500, 5,000 (LIME's default; the primary budget) and 10,000. 5 repeats. 10 reported features. Top-k for stability is 5. Settings with N ≤ d (only 500 at d = 500) are flagged as underdetermined and kept out of trend lines. |
| Seeds | Derived **per case** from base seeds (repeats 42–46, held-out 10042, references 20042/20043) with NumPy's `SeedSequence`, so cases are independent. Within a case, every model, d and N share the seed, so comparisons are paired. |
| LIME settings (explicit) | `lime==0.2.0.1`, `discretize_continuous=True`, `discretizer="quartile"`, `kernel_width=None` (LIME default `0.75·sqrt(d)`), `feature_selection="auto"` (resolves to `highest_weights` at 10 features), `sample_around_instance=False`, euclidean distance, Ridge surrogate with `alpha=1`. |
| Runtime fairness | **All models run single-threaded:** `n_jobs=1`, and the BLAS/OpenMP thread variables set to 1. **All models run interleaved in one process** as a randomized complete block design: each block (case × d × N × repeat) runs every model once in shuffled order; blocks run in seeded random order. Untimed warm-ups per model × d; garbage collection disabled inside timed calls; 11 speed probes record drift. |
| Model seeds | `random_state=42` on every model. Model fingerprints (hash of test probabilities) prove the same fitted models are used across variants. |
| Runtime definition | Total wall-clock time of `explain_instance()` (`perf_counter`): perturbation, model probability calls, distances and surrogate fitting. Excludes model training, warm-ups, explainer construction and quality scoring. Not split by stage. |
| Fidelity and error | Kernel-weighted **R²** and **RMSE** of the surrogate on 5,000 held-out perturbations per case × d, drawn exactly as LIME draws them and shared by every budget, repeat and model. LIME's own `explanation.score` (in-sample, inflated at small N) is kept as `fidelity_insample` for comparison only. Cases where the model's output barely moves (kernel-weighted SD < 0.01) are flagged `flat_output` and left out of R²; RMSE is kept. |
| Convergence | Top-5 Jaccard against two N = 50,000 reference explanations per case × d × model. Their agreement with each other is the **ceiling**; agreement relative to the ceiling is also reported. A reference is valid only if the two share on average ≥ 4 of 5 top features (Jaccard ≥ 0.667), a rule set before results. |
| Stability | For each case × d × budget, all 10 pairs of the 5 repeats compared with top-5 Jaccard. Spearman was dropped on 20-10-2026 (see `log_entries/20-10-2026.md`). |
| Statistics | Metrics averaged within each case, then a class-stratified case bootstrap (2,000 resamples) gives 95% CIs. Model comparisons are paired by case and tested with a sign-flip permutation test. A **primary family** (model pairs × d at N = 5,000 for runtime, held-out RMSE and reference Jaccard; ≤ 36 tests) is Holm-corrected at α = 0.05; everything else is secondary. Runtime scaling exponents come from power-law fits. |
| Sensitivity checks | `ranked_subset` (top-d features by ANOVA F on training labels) and `fixed_kernel` (width fixed at `0.75·sqrt(300)` ≈ 12.99). A d-trend that holds in all three variants is not caused by random subsets or kernel width. |
| Environment | `requirements-pinned.txt` (Python 3.13). Full runs stop if installed versions differ. |
| Hyperparameters | Unchanged from the notebooks except for `n_jobs` and `random_state`. Models are **not** tuned or matched to each other: the comparison uses a shared evaluation setup, not equally optimised models. |
| Dataset scope | UCI HAR only (Covertype removed). All dataset-specific values are in one block of `config.py`. All features are continuous, so there are no categorical features in the explainer. |

Model definitions (in `config.py`):

- **XGBoost:** `XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)`
- **Logistic Regression:** a `Pipeline` that standardises all 561 columns, followed by `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)`. Its `n_jobs` setting has no effect with lbfgs, so the thread environment variables keep it single-threaded.
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)`

## Folder layout

```
lime_study/
  README.md                  how to run; what each file does
  requirements-pinned.txt    exact package versions (checked at the start of every run)
  __init__.py                forces single-threaded BLAS/OpenMP before numpy loads
  threads.py                 the thread environment-variable helper
  config.py                  ALL shared constants, LIME settings, variants and model factories
  protocol.py                data, case selection, subsets, interleaved LIME benchmark, held-out
                             scoring, references, validation, version check, saving
  analysis.py                case-level aggregation, bootstrap CIs, sign-flip tests, Holm,
                             reference check, runtime scaling fits
  models/run_*.py            debug only: run one model on its own (not comparable)
  run_all.py                 runs all models together in one process per variant, then plots
  plot_results.py            reads results/, checks runs and variants match, writes figures/
  results/                   per-model output files (see below)
  figures/                   final figures and combined tables
  notes/                     this plan, design decisions, fixes, dataset justification
```

**One rule:** model scripts contain no data, LIME or sweep logic. Everything shared lives in `config.py`, `protocol.py` and `analysis.py`, so the models cannot drift apart.

### Files saved for each model (`results/`; variants in `results/<variant>/`)

| File | Contents |
|---|---|
| `<Model>_raw.csv` | One row per explanation (3,000 rows): case, true/predicted/explained class, d, N, repeat, seed, run and block position, runtime, in-sample and held-out R², held-out RMSE, the model's local output SD and `flat_output`, reference Jaccard and ceiling, and the feature weights as JSON |
| `<Model>_stability.csv` | One row per repeat pair (6,000 rows): top-5 Jaccard |
| `<Model>_references.csv` | One row per reference explanation (240 rows): seed, runtime, fidelity, ceiling, weights |
| `<Model>_summary.csv` | One row per d × N (20 rows): case-level means with 95% bootstrap CIs for every metric, `flat_cases`, relative reference agreement, `reference_valid`, `underdetermined` |
| `<Model>_performance.json` | Test-set accuracy, balanced accuracy, and macro precision/recall/F1 (2,947 test rows) |
| `<Model>_run_meta.json` | `run_id`, models in the run, version check, model fingerprint, explained cases, timestamp, machine/CPU, Python and package versions, data source/DOI/archive SHA-256 and subject IDs, thread settings, config snapshot, model parameters, fit time, benchmark time, speed-probe trace |

### Figures and tables (`figures/`; variants in `figures/<variant>/`)

- `lime_scaling_grid.png/.pdf`: runtime, held-out R² (flat cases excluded) and held-out RMSE against N, one column per d, one line per model, 95% CI bands. N ≤ d points are hollow and kept out of lines. All axes are linear.
- `lime_stability_grid.png/.pdf`: reference Jaccard (dashed line = reference ceiling, dotted if the reference fails its rule), relative reference agreement, and repeat Jaccard.
- `cost_vs_quality.png/.pdf`: mean runtime against reference Jaccard, one line per model, points ordered by budget.
- `paired_differences.csv`: model A − model B per metric, d and N, with CIs, sign-flip p-values, `p_holm_primary` and `p_holm_all`.
- `reference_check.csv`: reference ceiling, reference Jaccard and repeat Jaccard at each d and their changes between d values (Holm-corrected).
- `runtime_scaling.csv`: scaling exponents in N and d with CIs, runtime multiplier per doubling, and paired exponent differences between models.
- `machine_drift.csv`, `combined_summary.csv`, `performance_table.csv`.

## How to run

From the repository root, in a Python 3.13 environment with the pinned packages:

```bash
python -m pip install -r lime_study/requirements-pinned.txt
python -m lime_study.run_all --quick                                    # smoke test → results/quick, figures/quick
python -m lime_study.run_all --variants main ranked_subset fixed_kernel # full study, all variants
python -m lime_study.run_all --plots-only --variants main               # redraw from saved CSVs
```

Run all variants on the same machine. Keep it awake and plugged in, and close other heavy programs, because runtime is one of the measurements. The machine is recorded in `run_meta.json` automatically. The full run is roughly 1–1.5 hours per variant on the Mac (an estimate).

## Checks

Before saving, each run asserts:

- the environment matches `requirements-pinned.txt` (full runs);
- 3,000 raw rows, 6,000 stability rows, 240 reference rows and 20 summary rows per model;
- 30 cases and 150 runs in every group;
- every model explained the true class;
- no missing held-out scores;
- the rebuilt surrogate reproduces LIME's own local prediction for every explanation.

`load_har()` also asserts the train/test shapes, the class names, and that no volunteer is in both sets. Before plotting, `plot_results.py` checks that all models come from one run (one `run_id`), that every model explained the same cases, and that model fingerprints and cases match across variants.

Sanity checks to do by eye after a full run:

- runtime exponents are near or below 1 with tight CIs, and the scaling fit R² is high;
- held-out R² rises (or levels off) with N, and repeat and reference Jaccard rise with N;
- every reference passes its rule at most d, or the failures are reported;
- `machine_drift.csv` stays near 1.0;
- test accuracy is around 0.9 or higher for every model;
- re-running gives identical quality results and only runtimes change.

## Changes from the old experiments

- LIME constructor and surrogate settings are passed explicitly instead of relying on library defaults.
- `n_jobs=1` and single-threaded BLAS for every model. Previously XGBoost used 4 threads and the other models used library defaults.
- `random_state=42` has been added to Logistic Regression.
- Pairwise stability rows and per-explanation feature weights are now saved, not thrown away.
- Package versions, machine details and the full config are saved with every run.
- One shared `predict_proba` wrapper (numpy rows → named DataFrame) for every model.
- The exploratory cells (single-row demos, top-5/5 overlap, explainer-construction timing, full-data splits) have been dropped.
- **21-10-2026:** held-out quality measures, references with a validity rule, true-class explanations, interleaved randomized blocks, case-level statistics with Holm correction, runtime scaling fits, sensitivity variants, 5 cases per class, d = 50, per-case seeds, model fingerprints and a version check. See [`claim_safe_fixes.md`](claim_safe_fixes.md).

## Limitations to state in the paper

- One dataset (UCI HAR).
- The LIME background is 2,000 of the 7,352 training rows.
- The 30 explained cases are balanced by class, unlike the test set, so means are over this balanced set. With 5 per class, bootstrap intervals are still somewhat too narrow.
- Runtime is total explanation time, not split by stage. It includes the cost of building the DataFrame in the shared prediction wrapper, which is the same for every model.
- LIME samples each perturbed feature independently, so perturbations break the strong correlations between HAR's features (many are statistics of the same signal) and produce unrealistic rows, increasingly so as d grows.
- Features outside the perturbed subset are held at the case's values, so each d explains a slightly different function.
- Main-design subsets are random (seeded), so a small subset may contain few of the model's most important features; `ranked_subset` checks this, but ranked subsets are redundant.
- Kernel width scales with d (LIME default); `fixed_kernel` checks this.
- R² levels across models reflect how linear and how confident each model is locally. Compare trends, not levels.
- Model hyperparameters are not tuned or matched, so runtime differences describe these settings, not model families.
- When a model misclassifies a case, its explanation is of the true class, not its prediction.
- The flat-output threshold (0.01) and the reference rule (4/6) are judgement calls.
