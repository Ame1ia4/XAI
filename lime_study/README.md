# lime_study: controlled LIME cost/quality benchmark

This folder holds the controlled experiment for the paper *LIME Cost/Quality Scaling on High-Dimensional Tabular Data across multiple models*. It measures LIME **runtime**, **held-out fidelity and error** (the surrogate's kernel-weighted R² and RMSE on fresh perturbations it was not fitted on), **convergence** (top-5 Jaccard against two 50,000-sample reference explanations) and **stability** (top-5 Jaccard across repeated runs) for XGBoost, Logistic Regression and Random Forest on **UCI HAR** (561 features, 6 classes, official subject-based train/test split). The sweep covers 50, 100, 300 and 500 perturbed features, 500 to 10,000 LIME samples, and 30 explained test rows (5 per class). The dataset choice and its sources are in [`notes/dataset_justification.md`](notes/dataset_justification.md).

The full methodology, the decisions behind it and the list of changes from the old notebooks are in [`notes/lime_study_rebuild_plan.md`](notes/lime_study_rebuild_plan.md). Every design parameter and its rationale is in [`notes/design_decisions.md`](notes/design_decisions.md), and every fix made on the `claim-safe-fixes` branch, with sources, is in [`notes/claim_safe_fixes.md`](notes/claim_safe_fixes.md). Only results from this folder should go into the paper. The notebooks in `experiments/` are kept for reference only.

## Run

From the repository root:

```bash
pip install -r lime_study/requirements-pinned.txt
python -m lime_study.run_all --quick      # smoke test → results/quick, figures/quick
python -m lime_study.run_all              # full study (3,000 timed explanations per model, all models interleaved)
python -m lime_study.run_all --models RandomForest   # a subset of models
python -m lime_study.run_all --plots-only            # redraw figures from saved results
python -m lime_study.models.run_xgboost              # debug only: one model on its own (not comparable)
python -m lime_study.run_all --variants ranked_subset fixed_kernel   # sensitivity checks
```

Full runs stop at once if the installed packages or Python version differ from `requirements-pinned.txt` (`--allow-version-mismatch` overrides this for non-paper runs; quick runs only warn). The first run downloads the UCI HAR archive (~60 MB) into `lime_study/data/`, which is gitignored. Close other heavy programs during a full run, because runtime is one of the measurements.

## Files

| File | Role |
|---|---|
| `config.py` | **Every shared value:** seeds, dataset source, LIME background size, feature counts, budgets, repeats, explicit LIME settings, and the three model factories. Edit settings here and nowhere else. |
| `protocol.py` | UCI HAR download/loading (with shape and subject-overlap checks), case selection, nested feature subsets, the LIME benchmark loop, stability metrics, summaries, validation, and saving. Shared by every model. |
| `models/run_*.py` | Debug entry points that run one model alone. Their results get their own `run_id` and can't be mixed with an interleaved run. |
| `run_all.py` | Runs all models together in one subprocess (interleaved), then calls `plot_results`. |
| `analysis.py` | Case-level aggregation, class-stratified bootstrap CIs, sign-flip tests with Holm correction, the reference check, and runtime scaling fits. |
| `plot_results.py` | Builds the figures and combined tables from `results/`. |
| `threads.py`, `__init__.py` | Pin BLAS/OpenMP to one thread before numpy loads, so runtimes are comparable across models. |

## Outputs

- `results/<Model>_raw.csv`: one row per explanation, with the explained class (always the true class), runtime, run and block position, in-sample and held-out R², held-out RMSE, the model's local output SD and `flat_output` flag, Jaccard against the references and the reference ceiling, seed and feature weights.
- `results/<Model>_stability.csv`: one row per pair of repeats, with top-5 Jaccard.
- `results/<Model>_references.csv`: the two large-N reference explanations per case × feature count, with their own fidelity and agreement with each other (the ceiling for reference Jaccard).
- `results/<Model>_summary.csv`: one row per feature count × budget, with case-level means and 95% bootstrap CIs, `flat_cases`, reference agreement relative to the ceiling, participant-clustered `_cluster` CIs, the number of `participants`, `target_sd_mean` (the model's local output spread, the scale for RMSE), and a `low_sample_ratio` flag for N ≤ d.
- `results/<Model>_performance.json`: held-out accuracy, balanced accuracy and macro P/R/F1.
- `results/<Model>_run_meta.json`: `run_id` (shared by all models in one run), the version check, a model fingerprint (hash of test-set probabilities), the explained cases, machine, package versions, thread settings, full config, model parameters, timings and the speed-probe trace.
- `figures/lime_scaling_grid.{png,pdf}`: runtime, held-out R² (flat cases excluded) and held-out RMSE against budget, one column per feature count, with bootstrap CI bands. N ≤ d settings are hollow markers outside the lines. All axes are linear.
- `figures/lime_overview.{png,pdf}`: the headline figure, with one row each for runtime, fidelity (held-out R²) and stability (top-5 Jaccard between repeats) against budget, and one column per feature count.
- `figures/lime_stability_grid.{png,pdf}`: reference Jaccard (dashed line = reference ceiling), reference Jaccard relative to the ceiling, and repeat Jaccard.
- `figures/cost_vs_quality.{png,pdf}`: reference Jaccard against runtime.
- `figures/paired_differences.csv`: model A − model B per metric, feature count and budget, computed case by case, with bootstrap CIs, sign-flip p-values, `p_holm_primary` (pre-specified primary family) and `p_holm_all` (whole table).
- `figures/reference_check.csv`: per model, the reference ceiling, reference Jaccard and repeat Jaccard at each d and their changes between d values, at the primary budget, with Holm-corrected p-values.
- `figures/runtime_scaling.csv`: power-law exponents of runtime in N and d (overall, within each d, within each N, with an interaction term), the runtime multiplier per doubling, and paired exponent differences between models, all with bootstrap CIs.
- `figures/machine_drift.csv`: speed-probe trace through the timed phase (`relative_to_first`, `relative_to_median`). Three unrecorded probe runs come first, so the first probe is not a cold start.
- `figures/combined_summary.csv`, `figures/performance_table.csv`.
- Sensitivity variants write the same files to `results/<variant>/` and `figures/<variant>/`.

## Controls against bias

- **Same models and cases everywhere:** within a run, every condition uses one set of fitted models and the same explained rows. Across variants (separate runs), plotting checks the model fingerprints and explained rows match, and stops if not.
- **Independent rows, paired conditions:** every random seed (repeat, held-out set, references) is derived per explained row, so rows don't share random streams. Within a row, the same seed is used for every model, d and N, which keeps those comparisons paired.
- **Same explained class:** every model explains the case's true class, so all models explain the same quantity. `correct` records whether each model predicted it.
- **Run order:** all models run in one process as a randomized complete block design. Each block (case × d × N × repeat) runs every model once, in random order; blocks run in seeded random order. Drift hits all models alike, and paired runtime differences come from runs seconds apart. Untimed warm-ups run per model × d, automatic garbage collection is switched off inside each timed call (collected every 100 calls outside timing), and 11 speed probes trace the machine's speed.
- **Quality is scored out of sample:** `fidelity_heldout` (R²) and `heldout_rmse` use 5,000 fresh perturbations per case × feature count, drawn and weighted exactly as LIME does it, shared by every budget, repeat and model. LIME's in-sample `explanation.score` is kept as `fidelity_insample` for comparison only, because small N inflates it.
- **Flat outputs:** if the model's probability barely moves near a case (kernel-weighted SD < 0.01), the case is flagged `flat_output` and its R² is left out (R² divides by that movement). RMSE is still reported for every case.
- **Convergence, not just repeatability:** `reference_jaccard` compares each explanation with two 50,000-sample reference runs whose seeds are disjoint from the repeat seeds. How well the two references agree (`reference_ceiling`) is reported and tested across d in `reference_check.csv`, and `reference_jaccard_relative` divides by it. Only claim LIME is less stable at larger d if the relative agreement also falls.
- **Reference validation rule (set before results):** for each model × d, the two references must share on average at least 4 of their top 5 features (mean top-5 Jaccard ≥ 0.667, `reference_valid`). Where they don't, agreement with the reference isn't read as convergence: those comparisons leave the primary family, and the figure draws that ceiling dotted.
- **N ≤ d** (only N=500 at d=500 in the full sweep) is flagged `low_sample_ratio` and kept out of trend lines. LIME's ridge feature selection is regularised, so this is a low sample-to-feature ratio, not strictly underdetermined.
- **Participants:** the 30 cases come from the 9 test participants, at most 4 each (`MAX_CASES_PER_SUBJECT`), chosen by going round the classes and picking the least-used participant. `subject` is saved in `raw.csv`.
- **RMSE is absolute error** in probability units, and it is bounded by how much the model's output moves locally (`target_sd`). A lower RMSE for one model can just mean its output moves less, so read RMSE next to `target_sd_mean` and R².
- **Uncertainty:** cases are the unit of the tests. Repeats are averaged within a case, then a class-stratified bootstrap over cases (2,000 resamples, seeded) gives 95% CIs. Model comparisons are paired by case and tested with a sign-flip permutation test. Cases from one participant may be correlated, so p-values are conditional on these 9 participants; every summary and paired difference also has a participant-clustered bootstrap interval (`_cluster`), which is the conservative check.
- **Multiple comparisons:** the primary family (all model pairs × d at N=5,000, LIME's default, for runtime, held-out RMSE and reference Jaccard: up to 36 tests, fewer if a reference fails its rule) is Holm-corrected and is what the paper should claim. Everything else is secondary, with Holm over the whole table.
- **Runtime scaling (RQ1):** `runtime_scaling.csv` fits log runtime = a + b·log N + c·log d, so b and c are scaling exponents (1 = linear). Five budgets and four feature counts give 20 design points.
- **Sensitivity checks:** `ranked_subset` uses model-agnostic informative subsets (ANOVA F on training labels), so the d-axis doesn't depend on how much signal a random subset happens to contain. `fixed_kernel` fixes the kernel width at LIME's d=300 default, so the d-axis doesn't also change the neighbourhood size. A d-trend that holds in all three variants is not caused by either confound.
- **What stays model-dependent:** R² levels depend on how linear each model is locally, so compare fidelity *trends* across models, not levels. Runtime differences are for these exact model settings, not model families.

Each run checks the expected row counts and that every model explained the true class before saving. Plotting fails if the models' results come from different runs, or if a model's fingerprint or explained rows differ between variants.

## Adding a model or dataset

- **Model:** add a `build_<name>()` factory and a `MODEL_BUILDERS` entry in `config.py`, a style entry in `plot_results.MODEL_STYLE`, and optionally a debug `models/run_<name>.py`.
- **Dataset:** all dataset-specific values are in the "Dataset block" of `config.py`, with the loader in `protocol.load_har`. Covertype was dropped on 20-10-2026; its old results are in git history (commit `58c9cb4`).

## Troubleshooting

If plotting fails with `Failed to find font DejaVu Sans`, your matplotlib font cache is stale. Delete `~/.matplotlib/fontlist-*.json` and run again.

If the download fails with `CERTIFICATE_VERIFY_FAILED`, install the pinned requirements. The loader uses `certifi`'s certificates, because python.org's macOS Python ships without them.
