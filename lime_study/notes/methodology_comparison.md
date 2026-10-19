# Methodology comparison: intended design vs current implementation

## Scope

This note compares the experiment design provided in the project brief with the code currently present in `experiments/`. It is a read-only review; no experiments were run and no notebook outputs were generated.

The current shared implementation is `experiments/controlled_lime_protocol.py`. Each of the four model notebooks has a final controlled benchmark call to that helper: Logistic Regression, Random Forest, XGBoost, and TabFM. The older notebook sections remain exploratory and do not all follow the shared protocol.

## At-a-glance status

| Design item | Intended methodology | Current implementation | Status |
|---|---|---|---|
| Dataset | Covertype, about 580,000 rows, 54 predictors, 7 classes | `fetch_covtype`; 581,012 rows, 54 predictors, 7 classes | Implemented; exact full-data row count is 581,012 |
| Model training data | Fixed 80/20 split, seed 42 | First take a stratified 10,000-row sample with seed 42, then stratified 80/20 split with seed 42 (8,000 train, 2,000 test) | Partially aligned; split is fixed, but training does not use all 581,012 rows |
| LIME dimensionality axis | 18, 36, 54 features | Shared nested subsets of 18, 36, 54 features; first 10 numeric features always included, binary indicators added using one seeded permutation | Implemented |
| Perturbation budgets | 500, 1,000, 2,500, 5,000, 10,000 | Same five `num_samples` values | Implemented |
| LIME background | 50,000 observations | `BACKGROUND_SIZE = 2_000`; helper samples at most 2,000 from training data | Not aligned |
| Explanations per run | 10 features | `NUM_FEATURES = 10` | Implemented |
| Repeats and seeds | 5 runs with different explainer seeds for each budget | 5 runs; seed is 42 + repeat index (42–46), for each case, dimension, and budget | Implemented |
| Models | Logistic Regression, Random Forest, XGBoost, TabFM | Each notebook fits its model and calls the shared benchmark helper | Code is present for all four; successful controlled execution is not established by saved notebook outputs |
| LIME version | 0.2.0.1 | Notebook installation cells are unpinned; some saved package output reports 0.2.0.1, but no requirements/lock pins it | Not reproducibly pinned |
| Kernel width | Explicitly state default `0.75 · sqrt(d)` | Not passed explicitly; current LIME default is used, based on the number of columns given to each explainer | Implicit, not explicit |
| Continuous discretization | `discretize_continuous=True`, quartile discretizer | Not passed in shared helper; these are the `LimeTabularExplainer` defaults | Implicit, not explicit |
| Feature selection | Fix and state method | Not passed; `feature_selection='auto'` default applies. With 10 reported features, LIME takes its `highest_weights` branch, not forward selection | Implicit, not explicit |
| Surrogate regressor | Ridge, alpha = 1 | Not passed to `explain_instance`; LIME's default is Ridge with alpha 1 | Implicit, not explicit |
| Runtime | Wall-clock time per explanation, split by stage | One total `explain_instance()` duration per explanation | Partially aligned; stage timings are not implemented |
| Stability | Jaccard overlap of top-k features across seeds for the same instance | Shared helper calculates pairwise top-5 Jaccard plus Spearman per case, dimension, and budget | Implemented, with top-k fixed at 5 |
| Local fidelity | `explanation.score` or local prediction agreement | Stores `explanation.score` | Implemented for weighted R²; local prediction agreement is not stored |
| Saved results | Results for the controlled design | All four final controlled cells have no saved outputs in the checked notebook files | Not yet demonstrated |

## What the shared protocol does

### Data, split, and dimensionality

`load_controlled_covtype()` fetches the full dataset, shifts target labels from 1–7 to 0–6, draws a seeded stratified 10,000-row sample, then makes a seeded stratified 80/20 split. Every model notebook imports this helper. The resulting intended shapes are 8,000 training rows and 2,000 test rows.

The tested feature counts are implemented as nested subsets. All 10 numeric features are kept at each dimension, then 8, 26, or 44 binary indicator features are added for 18, 36, or 54 total features. A single seeded ordering of the indicators makes the smaller sets nested in the larger set. For each LIME perturbation, excluded features are held at the instance's observed values, and the complete 54-feature row is passed to the fitted classifier. Thus the current sweep measures the effect of changing LIME's perturbed feature dimension; the classifiers are not retrained at each dimension.

The helper samples three test cases per true class (21 cases) independently of the model predictions. This makes the explanation case set balanced by class even though the full Covertype test distribution is not. The model's predicted class is the label explained for each case.

### Budgets, repeats, and explanation length

The helper uses `num_samples` = 500, 1,000, 2,500, 5,000, and 10,000; 5 repeats; and 10 explanation features. The five explainer seeds are 42–46. The helper creates a fresh explainer for each repeat and uses a fixed 2,000-row training background (or fewer if fewer rows are available).

At 21 cases, this means 21 × 3 dimensions × 5 budgets × 5 repeats = **1,575 explanations per model**. Across all four models, a complete controlled run would contain **6,300 explanation records**.

### Metrics currently produced

- Runtime is measured with `perf_counter()` around `explain_instance()`. It includes perturbation creation, model probability calls, and surrogate fitting. It excludes model training, a prediction warm-up, background/case setup, and explainer construction.
- Fidelity is the returned `explanation.score`, the weighted local surrogate R². The helper does not store `local_pred` or calculate agreement between `local_pred` and the classifier's output.
- Stability compares repeats for each same case, dimension, and budget. The helper calculates pairwise Spearman correlation over the union of features with nonzero weights and top-5 Jaccard intersection-over-union. Five repeats produce 10 pairs per group.
- The returned summary includes mean/median/std runtime, mean/std fidelity, mean stability values, case count, explanation run count, and the fraction of selected cases predicted correctly.
- Predictive accuracy, balanced accuracy, macro precision, macro recall, and macro F1 are calculated separately on the 2,000-row test split.

The pairwise stability rows are aggregated into the summary and are not returned as a separate raw table. The raw CSV from each notebook contains per-explanation runtime/fidelity observations; the summary CSV contains aggregate stability means.

## LIME settings: implementation details and reproducibility

In the controlled helper, `LimeTabularExplainer` explicitly receives the background data, feature names, categorical indicator positions/names, classification mode, and random state. It does **not** explicitly receive `kernel_width`, `discretize_continuous`, `discretizer`, or `feature_selection`. The `explain_instance()` call does not explicitly receive `model_regressor` or `feature_selection` either.

These omissions currently invoke the documented upstream defaults: kernel width `0.75 * sqrt(number of explainer columns)`, continuous discretization enabled with quartile bins, `feature_selection='auto'`, and a default Ridge surrogate with alpha 1. Since the number of explainer columns changes across the 18/36/54 conditions, the default kernel width changes with that dimension. With `num_features=10`, the `auto` selector uses `highest_weights`; LIME only uses forward selection when the requested explanation has six or fewer features. For reproducibility, record these settings explicitly in the benchmark setup and capture the installed LIME version.

The `python3` environment available during this review cannot import `lime`. Notebook outputs contain package-install logs reporting LIME 0.2.0.1, but the notebook install cells do not pin that version. The environment used for a real run therefore still needs to be confirmed and recorded.

## Main differences and open decisions

1. **Background size is different.** The brief specifies 50,000 rows passed to `LimeTabularExplainer`; the shared protocol uses 2,000. Older notebook sections also use varying backgrounds. The experiment owner should settle which background size is canonical before publishing results. If 50,000 is required, update the shared helper consistently for all models and feature dimensions.
2. **Runtime is not split by stage.** The implementation produces one end-to-end LIME duration. It does not separately time sampling, model prediction, distance/kernel weighting, and surrogate fitting. Those internal stages are intertwined in `explain_instance()`; stage-level timing would require instrumentation or a carefully defined profiling method. Until implemented, results should be described as total explanation wall-clock time.
3. **Model split scope is smaller than the dataset description can imply.** The full Covertype data are fetched, but the protocol benchmarks models trained on a 10,000-row stratified sample. Say so explicitly in the methodology. The XGBoost notebook repeats the 80/20 split in an older cell after the shared loader; it uses the already sampled 10,000 rows and same seed, but the redundant split should be removed or the helper should remain the sole split definition to prevent drift.
4. **Random state is not uniformly explicit across model definitions.** The data and LIME seeds are fixed at 42, and Random Forest/XGBoost set model seeds. Logistic Regression uses `lbfgs` and omits `random_state`; TabFM's visible setup does not establish a model seed. Document whether these are deterministic in the chosen environment and set/record model randomness consistently where applicable.
5. **Some old notebook sections use different stability calculations.** Treat only the final `benchmark_lime()` outputs as the controlled result. Earlier cells include single-case, full-dimension demonstrations and sometimes use top-5 overlap divided by 5 rather than Jaccard intersection-over-union. Those measurements should not be combined with controlled summary data.
6. **There are no saved controlled results yet.** The final benchmark cells in the four notebooks have empty output arrays in the checked files. Code calls are present, but this review found no saved evidence that the controlled benchmark has completed for any model. TabFM also has a setup cell that clones its repository and loads a large pretrained model, so its runtime environment and model availability need to be recorded separately.
7. **The 50,000-row requirement may refer to a LIME background, not the model training sample.** Keep these concepts distinct: `num_samples` (500–10,000) is the number of local perturbations generated per explanation; the background size describes data passed at explainer construction; and the current model-training sample is 10,000 rows before splitting.

## Recommendation for a reproducible method statement

Before final runs, agree and encode one canonical configuration for all four models:

- Stratified 10,000-row modeling sample and seeded 80/20 split, or an explicitly chosen alternative.
- Exact LIME training/background size (resolve 2,000 vs 50,000).
- 18/36/54 nested LIME feature subsets, with the same definition for every model.
- `num_samples` values 500, 1,000, 2,500, 5,000, and 10,000; 10 reported features; 5 explainer seeds (42–46).
- Explicit LIME settings: version 0.2.0.1, `kernel_width=None` (or formula explicitly stated as `0.75 * sqrt(d)`), `discretize_continuous=True`, `discretizer='quartile'`, `feature_selection='auto'` (which resolves to `highest_weights` at k=10), and Ridge `alpha=1`.
- Total explanation time as the current primary runtime outcome, unless a stage-profiling implementation is added and validated.
- Pairwise top-5 Jaccard and Spearman stability, and weighted local R² from `explanation.score`; optionally save `local_pred` and model probability at the explained point if local prediction agreement is also to be reported.
- Model-specific seeds/settings, dependency versions, execution machine, and saved raw/summary outputs for all four models.

## Source references

- `experiments/controlled_lime_protocol.py` — shared protocol constants and implementation.
- `experiments/CovTypeLogRegLimeExp.ipynb` — Logistic Regression controlled call and earlier exploratory benchmark.
- `experiments/randomForest.ipynb` — Random Forest controlled call and earlier exploratory benchmark.
- `experiments/coverType_XGBoost_activity.ipynb` — XGBoost model setup and controlled call.
- `experiments/Covertype_TabFM_XAI.ipynb` — TabFM setup and controlled call.
- Upstream LIME implementation documents the `LimeTabularExplainer` defaults and automatic feature-selection rule: [lime_tabular.py](https://github.com/marcotcr/lime/blob/master/lime/lime_tabular.py) and [lime_base.py](https://github.com/marcotcr/lime/blob/master/lime/lime_base.py).
