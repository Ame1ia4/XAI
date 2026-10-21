# Design decisions: what the study uses and why (21-10-2026)

This note records every design parameter of the controlled LIME study as of the `claim-safe-fixes` branch, with the reason for each value. Each value lives in `lime_study/config.py` unless stated otherwise. The fixes that led here are in [`claim_safe_fixes.md`](claim_safe_fixes.md). The dataset choice is in [`dataset_justification.md`](dataset_justification.md).

Each decision is marked as one of:

- **Literature:** a published method or convention supports it.
- **Default:** it matches LIME's own default, so results describe LIME as people actually use it.
- **Judgement:** a reasoned choice without a single authoritative source. State it in the paper as a choice, and where noted, check that results don't depend on it.

> **Sources:** references are listed in [`claim_safe_fixes.md`](claim_safe_fixes.md#references) and were cited from memory. Check them against the originals before citing.

## Study size at a glance (full sweep)

| Quantity | Value |
|---|---|
| Models | 3 (XGBoost, Logistic Regression, Random Forest) |
| Explained test rows (cases) | 30 (5 per class × 6 classes) |
| Perturbed-feature counts d | 50, 100, 300, 500 |
| Sample budgets N | 500, 1,000, 2,500, 5,000, 10,000 |
| Repeats per setting | 5 |
| Timed explanations | 3,000 per model, 9,000 in total |
| Repeat pairs (stability) | 6,000 per model |
| Reference explanations | 240 per model (30 cases × 4 d × 2), each N = 50,000 |
| Held-out perturbations | 5,000 per case × d |
| Variants | `main`, plus `ranked_subset` and `fixed_kernel` sensitivity checks |

---

## Data

| Decision | Value | Type | Why |
|---|---|---|---|
| Dataset | UCI HAR, 561 continuous features, 6 classes | Literature | High-dimensional, dense, real tabular data. See `dataset_justification.md`. |
| Train/test split | Official subject-based split: 21 volunteers / 7,352 rows train, 9 volunteers / 2,947 rows test | Literature | No volunteer is in both sets, so test rows are genuinely unseen (Anguita et al., 2013, in `dataset_justification.md`). Checked in code. |
| Models trained on | The full training split, all 561 features | Judgement | Models are not retrained per feature count, so d only changes what LIME perturbs, not the model. |

## Explained cases

| Decision | Value | Type | Why |
|---|---|---|---|
| Cases per class | 5 (30 total) | Literature + judgement | Cases are the independent units for every interval and test (Hurlbert, 1984). With n per class, the bootstrap variance is too small by (n − 1)/n: 0.67 at 3, 0.80 at 5 (Efron & Tibshirani, 1993). 5 balances this against runtime (~5/3 of the 3-per-class cost). The test set has 420–537 rows per class. |
| Selection | Seeded random sample per true class, made before any model is fitted | Literature | Selection can't depend on any model, so no model gets easier rows. Classes are balanced, unlike the test set; state that means are over a class-balanced set. |
| Class explained | The row's **true** class, for every model | Literature | All models must explain the same quantity for a controlled comparison. LIME explains one chosen class at a time (Ribeiro et al., 2016). `correct` records whether each model predicted it. |

## The d axis (perturbed features)

| Decision | Value | Type | Why |
|---|---|---|---|
| Feature counts | 50, 100, 300, 500 | Judgement + literature | 500 is TabFM's `max_num_features`, kept for comparability with the original plan. 50 adds a fourth point for estimating the runtime exponent in d (Goldsmith et al., 2007) and is never underdetermined. 100 and 300 fill the range. |
| Subsets | Nested prefixes of one seeded random permutation (50 ⊂ 100 ⊂ 300 ⊂ 500) | Judgement | Nesting means a larger d only adds features, so changes with d aren't caused by swapping features out. The same subsets are used for every model. |
| Features outside the subset | Held at the case's own values | Judgement | The model always sees all 561 features. Each d therefore explains a slightly different function (the model with other features fixed). State this in the paper. |
| Sensitivity check: `ranked_subset` | Top-d features by ANOVA F against the training labels | Literature | A standard model-free filter (Guyon & Elisseeff, 2003). It checks whether trends come from random subsets holding little signal. Every model still gets the same subsets. |

## The N axis (sample budgets) and repeats

| Decision | Value | Type | Why |
|---|---|---|---|
| Budgets | 500, 1,000, 2,500, 5,000, 10,000 | Default + judgement | They bracket LIME's default `num_samples=5000` (`lime_tabular.py`). The roughly doubling steps give a usable cost–quality curve. |
| Primary budget | 5,000 | Default | LIME's default, chosen before seeing results, so the main comparisons describe LIME as typically used. |
| Repeats | 5 per setting (seeds derived per case) | Judgement | 5 repeats give 10 pairs for stability per case × setting. Repeat-to-repeat variability is the quantity of interest (Visani et al., 2022). |
| Underdetermined settings | N ≤ d flagged, out of trend lines and the primary family | Literature | Ridge feature selection over d features from N ≤ d samples is underdetermined (Hastie et al., 2009, ch. 18). Only N = 500 at d = 500 qualifies. |

## LIME settings (all passed explicitly; `lime==0.2.0.1`)

| Decision | Value | Type | Why |
|---|---|---|---|
| Discretization | `discretize_continuous=True`, quartile discretizer | Default | LIME's tabular defaults, made explicit for reproducibility. |
| Kernel | Exponential, width `0.75·√d` | Default | LIME's default. Because it grows with d, it is part of what the d axis measures (Garreau & von Luxburg, 2020a, 2020b). |
| Sensitivity check: `fixed_kernel` | Width fixed at `0.75·√300 ≈ 12.99` | Judgement | Separates the kernel-width effect from d. d = 300 is the reference point, so that setting is unchanged. A fixed width is not neutral (distances grow with d), so this is a check, not the main design. |
| Feature selection | `"auto"`, which resolves to `highest_weights` for 10 features | Default | LIME's default. LIME only uses forward selection for 6 or fewer features. |
| Features reported | 10 | Judgement | From the original brief. |
| Surrogate | Ridge, `alpha=1`, intercept | Default | LIME's default surrogate, constructed explicitly. |
| Distance | Euclidean | Default | LIME's default. |
| Sampling | `sample_around_instance=False` | Default | LIME's default. |
| Background | 2,000 training rows (seeded) | Judgement | The brief asked for 50,000, but the training split has only 7,352 rows. The background is used to fit the quartile discretizer and sampling frequencies; 2,000 rows (~500 per quartile bin) estimates these precisely while keeping explainer construction cheap. State it as a limitation. |

## Models

| Decision | Value | Type | Why |
|---|---|---|---|
| Models | XGBoost (250 trees, depth 6, lr 0.1), Logistic Regression (standardised, lbfgs), Random Forest (100 trees) | Judgement | Three common model families of different complexity. Hyperparameters are carried over from the original notebooks and are **not** tuned or matched. Runtime differences describe these settings, not model families. |
| Seeds | `random_state=42` on every model | Literature | Reproducible fits. Verified across variants by model fingerprints. |
| Threads | `n_jobs=1` and all BLAS/OpenMP thread variables set to 1 | Judgement | Runtime is a measurement, so every model gets the same single core. |

## Quality measures

| Decision | Value | Type | Why |
|---|---|---|---|
| Held-out set size | 5,000 perturbations per case × d | Judgement | The same size as LIME's default budget. Large enough that the weighted R²/RMSE estimates are precise, and the set is shared by every budget, repeat and model. |
| Held-out metrics | Kernel-weighted R² and RMSE | Literature | LIME's own locality-weighted square loss (Ribeiro et al., 2016) scored out of sample (Hastie et al., 2009, ch. 7). Neighbourhood squared error is standard (Plumb et al., 2018). |
| Flat-output threshold | Kernel-weighted SD of the model's probability < 0.01 | Judgement | One percentage point. Below this, R² is unstable (Kvålseth, 1985), so R² is dropped and RMSE kept. Check that results hold at 0.005 and 0.02. |
| Stability metric | Top-5 Jaccard (intersection over union) | Literature + judgement | Set overlap of the most important features is a common stability measure (Nogueira et al., 2018, discuss its properties). Top-5 of 10 reported features focuses on the features a user would read first. Spearman was dropped on 20-10-2026 because its chance level differs across models. |
| References | 2 per case × d × model, N = 50,000, seeds separate from repeats | Judgement | 5× the largest budget, feasible in runtime (~4 s each at d = 500 on the Mac, estimated). Two references give a measure of the reference's own noise. |
| Reference validity rule | Mean reference self-agreement ≥ 4/6 ≈ 0.667 (≥ 4 of 5 features shared) | Literature + judgement | A comparison cannot show more agreement than the reference has with itself (Spearman, 1904; Nili et al., 2014). Top-5 Jaccard only takes 6 values, so "4 of 5 shared" is the natural cut-off. Set before any full results. |

## Statistics

| Decision | Value | Type | Why |
|---|---|---|---|
| Unit of analysis | The explained case; repeats averaged within case first | Literature | Avoids treating dependent measurements as independent (Hurlbert, 1984). |
| Confidence intervals | Class-stratified case bootstrap, 2,000 resamples, 95% percentile | Literature | Stratifying keeps classes balanced (Davison & Hinkley, 1997); ≥ 1,000 resamples for percentile intervals (Efron & Tibshirani, 1993). The same resamples are used for every setting and model (paired). |
| Model comparisons | Per-case differences; sign-flip permutation test (exact or 10,000 flips) | Literature | Valid for paired designs without assuming normality (Good, 2005); p never zero (Phipson & Smyth, 2010). |
| Multiple comparisons | Holm at α = 0.05 within a pre-specified primary family; Holm over all as secondary | Literature | Holm (1979); pre-specified primary family (Bender & Lange, 2001). |
| Primary family | Model pairs × d at N = 5,000, for runtime, held-out RMSE and reference Jaccard (≤ 36 tests) | Judgement | One cost metric and two quality metrics that stay valid when outputs are flat. Fixed in `config.py` before results. Reference comparisons drop out where a reference fails its rule. |
| Runtime scaling | log(runtime) = a + b·log N + c·log d, plus within-d / within-N slopes and an interaction model | Literature | Empirical complexity via power-law fits (Goldsmith et al., 2007). |

## Run procedure

| Decision | Value | Type | Why |
|---|---|---|---|
| Run order | All models in one process; randomized complete blocks (case × d × N × repeat), models shuffled within each block, blocks in seeded random order | Literature | Randomization against time trends (Fisher, 1935); blocking makes model comparisons drift-free and paired (Montgomery, 2017, ch. 4); benchmarking guidance (Mytkowicz et al., 2009; Kalibera & Jones, 2013). |
| Seeds | Derived per case with `SeedSequence`; shared within a case by every model, d and N | Literature | Independent cases for the tests; common random numbers within a case for paired comparisons (Law, 2015). |
| Warm-up | One untimed explanation per model × d before timing | Literature | Excludes first-call overhead (Georges et al., 2007). |
| Timed region | `explain_instance()` only: sampling, model calls, distances, surrogate fit. Excludes training, explainer construction and quality scoring | Judgement | Measures the per-explanation cost a user pays. Not split by stage (stated limitation). |
| Garbage collection | Disabled inside each timed call, collected every 100 calls outside timing | Literature | Collection pauses are timing noise (Georges et al., 2007). |
| Drift monitoring | 11 speed probes through the timed phase (`machine_drift.csv`) | Judgement | Shows whether the machine changed speed during the run. |
| Environment | `requirements-pinned.txt` (Python 3.13); full runs stop on any mismatch | Judgement | Library versions change numerics and speed, so results are only reproducible in the pinned environment. |
| Cross-variant check | Model fingerprints and explained cases must match across variants | Literature | Paired comparisons need the same units and treatments (Montgomery, 2017). |

## Known limitations of these choices

- One dataset (UCI HAR).
- 30 cases: bootstrap intervals are still somewhat too narrow (factor 0.80 on the variance).
- LIME perturbs each feature independently, so perturbed rows get less realistic as d grows (HAR's features are strongly correlated).
- Features outside the subset are fixed at the case's values, so each d explains a slightly different function.
- Ranked subsets are redundant, which can lower stability independently of d.
- The background is 2,000 rows, not the 50,000 in the original brief.
- Runtime is total explanation time, not split by stage, and includes the shared DataFrame wrapper.
- R² levels across models reflect local linearity and confidence. Compare trends, not levels.
- Model hyperparameters are not tuned or matched, so runtime differences describe these settings, not model families.
- The flat-output threshold (0.01) and reference rule (4/6) are judgement calls.
