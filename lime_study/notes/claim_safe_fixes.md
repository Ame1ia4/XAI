# Fixes on the `claim-safe-fixes` branch (21-10-2026)

This note lists every change made on the `claim-safe-fixes` branch, why each was needed, and the evidence or literature behind it. The design parameters themselves (feature counts, sample sizes, thresholds and so on) are justified in [`design_decisions.md`](design_decisions.md).

**Status:** the code is complete and passes a quick smoke test (all three models, two variants). The full study has **not** been run with this code yet. Results saved before this branch (`results/*.csv` from 20-10-2026) use the old design and must not go into the paper. Rerun with:

```bash
python -m lime_study.run_all --variants main ranked_subset fixed_kernel
```

> **Sources:** the references at the end were cited from memory. Check authors, venues and page numbers against the originals before citing them in the paper.

## Why this branch exists

The study asks three questions:

1. How does LIME's runtime per explanation scale with N (perturbation samples) and d (perturbed features)?
2. How does explanation quality change with N and d?
3. How does the black-box model change the cost/quality trade-off?

A review of the earlier `lime_study` code found it well controlled for question 1 but confounded for questions 2 and 3, mainly because of how quality was measured. Every fix below closes a specific way in which a result could have been wrong or misleading.

## Summary

| # | Problem | Fix | Question affected |
|---|---|---|---|
| 1 | Fidelity (`explanation.score`) was measured on the surrogate's own training samples, so small N looked better | Held-out R² and RMSE on 5,000 fresh perturbations | 2, 3 |
| 2 | Stability between repeats only shows repeatability, not convergence | Agreement with two 50,000-sample reference explanations | 2, 3 |
| 3 | References can themselves be noisy, especially at high d | Reference ceiling, relative agreement, pre-set validity rule | 2 |
| 4 | R² is unstable when the model's output barely changes | `flat_output` flag; R² dropped for flat cases; RMSE kept | 2, 3 |
| 5 | Settings with N ≤ d are underdetermined | Flagged and kept out of trend lines | 2 |
| 6 | Models sometimes explained different classes for the same row | Every model explains the true class | 3 |
| 7 | Models ran one after another, so machine drift could favour one | All models interleaved in one randomized block design | 1, 3 |
| 8 | Uncertainty treated dependent repeats as independent | Case-level bootstrap CIs, paired by case | all |
| 9 | 225 confidence intervals with no correction | Sign-flip tests, Holm correction, small primary family | 3 |
| 10 | Runtime scaling was read off a graph | Power-law fits giving scaling exponents | 1 |
| 11 | Random feature subsets and LIME's kernel width both change with d | Two sensitivity variants: `ranked_subset`, `fixed_kernel` | 2 |
| 12 | 3 cases per class is too few for reliable intervals | 5 cases per class (30) | all |
| 13 | Runtime exponent in d fitted from only 3 points | d = 50 added | 1 |
| 14 | Variants refit models; nothing proved they matched | Model fingerprints checked across variants | all |
| 15 | All cases shared the same random seeds | Seeds derived per case | all |
| 16 | Environment not checked against pinned versions | Version check; full runs stop on mismatch | all |
| 17 | `gc.collect()` before every timed call cost ~6 min per variant | Garbage collection disabled inside timed calls instead | 1 |
| 18 | Small bugs | Plot crash, speed-probe warm-up, threshold rounding, pandas warning | — |

---

## 1. Held-out fidelity and error

**Problem.** LIME's `explanation.score` is the surrogate's weighted R² on the same samples it was fitted on, after its 10 features were also chosen on those samples. In-sample error underestimates true error, and the gap is largest when there are few samples per parameter (Hastie et al., 2009, ch. 7). Small N therefore scored artificially well. The old results showed this directly: Logistic Regression's fidelity at d = 100 *fell* from 0.137 to 0.085 as N rose from 500 to 10,000.

**Fix.** For each case × d, 5,000 fresh perturbations are drawn with LIME's own sampler, scaling, distance and kernel (`protocol.holdout_draw`). Every fitted surrogate is scored on them:

- `fidelity_heldout`: kernel-weighted R²;
- `heldout_rmse`: kernel-weighted root mean squared error, in probability units.

The same held-out set is used for every budget, repeat and model, so differences between them never come from different test samples. A built-in check stops the run if the rebuilt surrogate does not reproduce LIME's own `local_pred` at the explained row. LIME's score is kept as `fidelity_insample` for comparison only.

**Why weighted.** LIME defines fidelity as a loss weighted by proximity to the explained row (Ribeiro et al., 2016, Eq. 2). An unweighted score would judge the surrogate on distant samples it was never meant to fit.

**Why RMSE as well as R².** RMSE is the square root of LIME's own locality-weighted square loss, so it is the most direct measure of "how wrong is the explanation". Evaluating local explanations by squared error on neighbouring points is established practice (Plumb et al., 2018). R² is relative to how much the model's output varies locally (see fix 4). Reporting both shows absolute and relative error.

**Evidence.** In the smoke test, held-out R² *rises* with N (Random Forest, d = 300: 0.03 → 0.15) where the in-sample score fell.

## 2. Agreement with large-N reference explanations

**Problem.** Top-5 Jaccard between repeats measures *repeatability*. Five small-N runs can agree with one another and still all be wrong, for example by repeatedly selecting the same spurious features.

**Fix.** For each case × d × model, two reference explanations are run with N = 50,000 and seeds separate from the repeat seeds. `reference_jaccard` is each explanation's top-5 Jaccard with the references, averaged over both. This measures *convergence*: how close an explanation at a given N is to what LIME settles on with far more samples.

**Why.** LIME's sampling variability shrinks as N grows (Zhang et al., 2019; Visani et al., 2022), and the number of perturbations determines how reliable an explanation is (Slack et al., 2021). A large-N run is the natural estimate of where LIME converges.

## 3. Checking the references themselves

**Problem.** A 50,000-sample reference is still slightly noisy. If the references get noisier as d grows, agreement with them falls even when LIME is not getting worse.

**Fix.**

- `reference_ceiling`: the top-5 Jaccard between the two references (the most agreement one could expect). Drawn as a dashed line in `lime_stability_grid`.
- `reference_jaccard_relative`: agreement divided by the ceiling (ratio of means, bootstrapped).
- `reference_check.csv`: the ceiling, reference agreement and repeat agreement at each d, with tests of their changes between d values (Holm-corrected).
- **Validity rule, fixed before any full results:** for each model × d, the two references must share on average at least 4 of their top 5 features (mean top-5 Jaccard ≥ 4/6 ≈ 0.667). Where they don't, `reference_valid` is false, the ceiling is drawn dotted, and reference-agreement comparisons leave the primary family (fix 9). A tiny tolerance stops float rounding at exactly 4/6 from flipping the decision.

**Why.** Agreement with a reference cannot exceed the reference's agreement with itself. Psychometrics corrects for this as attenuation (Spearman, 1904); neuroscience calls the same bound a "noise ceiling" (Nili et al., 2014). Top-5 Jaccard only takes the values 1, 0.667, 0.429, 0.25, 0.111 and 0, so "4 of 5 shared" is the natural threshold.

**Rule for the paper:** only claim "LIME becomes less stable at higher d" if repeat Jaccard falls (it uses no reference) **and** relative agreement also falls.

**Evidence.** In the smoke test (5,000-sample references), XGBoost's ceiling was 0.78 / 0.83 / 0.67 across d, and Logistic Regression failed the rule at d = 300 and 500.

## 4. Flat model outputs

**Problem.** R² = 1 − error ÷ output variation. When a model is nearly certain everywhere around a case (probability ≈ 0.99), there is almost no variation to explain and R² becomes unstable or meaningless (Kvålseth, 1985). A test in sklearn gave R² ≈ −1 × 10²⁸ for a target varying only at the 10⁻¹⁷ level. R² levels across models then partly reflect how *confident* each model is.

**Fix.** `target_sd` is the kernel-weighted standard deviation of the model's probability over the held-out perturbations. If it is below 0.01 (one percentage point), the case × d is flagged `flat_output`, and its R² is excluded from summaries and comparisons. RMSE is reported for every case. Summaries report `flat_cases`.

**Why both measures.** RMSE alone rewards flat models, because an output that never moves is trivially easy to imitate. R² alone breaks when the output is flat. Together they give a fair picture.

**Evidence.** In the smoke test, 17% of XGBoost's and 11% of Logistic Regression's case × d combinations were flat (output SD as low as 0.0004); none of Random Forest's were.

## 5. Underdetermined settings (N ≤ d)

**Problem.** With `feature_selection="auto"` and 10 features, LIME uses `highest_weights`, which first fits a Ridge regression on all d features. With N ≤ d there are at least as many unknowns as samples, so the feature choice is mostly noise (Hastie et al., 2009, ch. 18). The old results showed this at N = 500, d = 500: Jaccard 0.01–0.03, fidelity 0.03–0.16.

**Fix.** Settings with `num_samples ≤ feature_count` are flagged `underdetermined`, drawn as hollow markers, kept out of trend lines, and excluded from the primary family. In the full sweep this is only N = 500 at d = 500.

## 6. Every model explains the same class

**Problem.** Each model explained the class it predicted. For test row 8838 (true class 2), Logistic Regression explained class 0 and the others class 2, so the "paired" comparison compared answers to different questions.

**Fix.** Every model explains the row's **true** class (`explained_class` column; the run stops if any model differs). `predicted_class` and `correct` are still recorded.

**Why.** LIME explains one chosen class at a time (the `labels` argument; Ribeiro et al., 2016). A controlled comparison must measure the same quantity under every model, and the true class is fixed before any model is fitted. When a model is wrong, its explanation answers "what evidence does the model see for the correct class?"; report how many rows that affects using `correct`.

## 7. Fair run order: randomized complete block design

**Problem.** Models ran one after another, each in its own process. Any slow change in the machine (heat, background tasks, power state) would be absorbed by whichever model ran during it.

**Fix.** All models now run in **one process** as a randomized complete block design (`protocol.benchmark_lime`):

- a *block* is one case × d × N × repeat;
- each block runs every model once, in a freshly shuffled order;
- blocks run in a seeded random order;
- `run_position` and `block_position` are recorded;
- 11 speed probes (a fixed numerical workload) trace the machine's speed through the run (`machine_drift.csv`);
- each run gets a `run_id`, and plotting refuses to mix results from different runs. The per-model scripts are now debug-only.

This replaces an intermediate version that shuffled order within each model and shuffled model order.

**Why.** Randomizing treatment order is the basic defence against hidden time trends (Fisher, 1935). Blocking goes further: the models' runs for the same setting happen seconds apart, so drift cancels out of paired model comparisons and they become more precise (Montgomery, 2017, ch. 4). Benchmarking research shows that setup and ordering can bias timings substantially and recommends randomization and repeated measurement (Mytkowicz et al., 2009; Kalibera & Jones, 2013).

**Trade-off.** Models no longer have separate processes, so one model's memory and cache use can affect the next. Random order within each block means every model follows every other equally often, so this effect is spread evenly rather than favouring one model.

## 8. Uncertainty from cases, paired across models

**Problem.** The old ±1 std bands treated 90 runs (or 180 repeat pairs) per setting as independent. The 5 repeats of one case share that case, and each repeat appears in 4 of the 10 pairs. Treating dependent measurements as independent makes uncertainty look smaller than it is (Hurlbert, 1984).

**Fix** (`analysis.py`). Metrics are averaged within each case first. A class-stratified bootstrap over cases (2,000 resamples, seeded) gives 95% percentile intervals. The same resamples are reused for every setting and model, so comparisons are paired.

**Why.** The case is the independent unit. Resampling within class keeps the design balanced (Davison & Hinkley, 1997). At least 1,000 resamples is the usual guidance for percentile intervals (Efron & Tibshirani, 1993). All models explain the same cases, so case-by-case differences remove between-case variation from model comparisons.

## 9. Multiple-comparison control

**Problem.** `paired_differences.csv` had 225 intervals (3 model pairs × 15 settings × 5 metrics). At 95%, about 11 would exclude zero by chance alone.

**Fix.**

- Each model difference is tested with a **sign-flip permutation test** on per-case differences: exact enumeration when there are few cases, otherwise 10,000 random sign flips with p = (b + 1)/(B + 1).
- A **primary family** is fixed in `config.py` before any results: every model pair × d at N = 5,000 (LIME's default) for runtime, held-out RMSE and reference Jaccard. That is up to 36 tests, fewer where a reference fails its validity rule (fix 3). It is **Holm-corrected** (`p_holm_primary`, `significant_primary`). This is what the paper should claim.
- Everything else is secondary, with Holm over the whole table (`p_holm_all`). Present it as descriptive.

**Why.**

- **Holm** controls the probability of *any* false positive in the family. It is uniformly more powerful than Bonferroni and makes no assumption about dependence between tests (Holm, 1979). That matters because the metrics are correlated.
- **A small pre-specified primary family** is the standard way to keep the main claims strongly supported while still reporting everything (Bender & Lange, 2001).
- **Sign-flip test:** with 30 cases a t-test's normality assumption is hard to defend. The sign-flip test only needs the two models to be interchangeable under the null, which holds because they are compared on the same cases (Good, 2005). The + 1 stops a p-value from being exactly zero (Phipson & Smyth, 2010).

## 10. Runtime scaling quantified (question 1)

**Problem.** "Runtime grows roughly linearly" was read off the plots.

**Fix.** `runtime_scaling.csv` fits log(runtime) = a + b·log N + c·log d per model. b and c are **scaling exponents** (1 = linear, below 1 = sublinear). `multiplier_per_doubling` = 2^exponent. Also reported: the N exponent within each d, the d exponent within each N, an interaction model (does the N exponent change with d?), fit R², and paired exponent differences between models (question 3). All with bootstrap 95% intervals.

**Why.** Fitting power laws on log–log axes is the established way to measure empirical computational complexity (Goldsmith et al., 2007). Every case has the same N and d values, so the fit equals OLS on the across-case mean of log runtime, which makes the case bootstrap exact and cheap. If the within-d or within-N exponents differ a lot, or the interaction is far from zero, report that one power law is too simple.

## 11. Sensitivity variants for the d axis

**Problem.** Two things besides d change with d:

- **Random subsets:** a random 100-feature subset may contain little of what drives the model, so quality across d also reflects how much signal fell inside the subset.
- **Kernel width:** LIME's default width is 0.75·√d, so changing d also changes the neighbourhood size. Explanations depend strongly on this width (Garreau & von Luxburg, 2020a, 2020b; Zhang et al., 2019).

**Fix.** Two variants (`--variants`), each written to `results/<variant>/` and `figures/<variant>/`:

- `ranked_subset`: subsets are the top-d features by ANOVA F against the training labels (`f_classif`). This is a standard filter method (Guyon & Elisseeff, 2003). It uses no model, so every model still gets the same nested subsets.
- `fixed_kernel`: kernel width fixed at 0.75·√300 ≈ 12.99 for every d (LIME's default at d = 300).

**How to read them.** If a d-trend holds in all three variants, neither confound explains it. A fixed width is not neutral either: distances grow with d, so it makes the neighbourhood relatively narrower at high d. That is why it is a check, not the main design. Ranked subsets are highly redundant in HAR, which can lower stability for reasons unrelated to d.

## 12. Five cases per class

**Problem.** With 3 cases per class, each bootstrap stratum had 3 members. The bootstrap variance of a mean is too small by a factor of (n − 1)/n, which is 0.67 at n = 3 (Efron & Tibshirani, 1993; Davison & Hinkley, 1997).

**Fix.** `FULL_SWEEP` explains 5 cases per class (30 cases). The factor becomes 0.80, the sign-flip tests gain power, and generalisation rests on more independent rows (Hurlbert, 1984). The HAR test set has 420–537 rows per class, so the data allow it. Runtime grows by about 5/3.

## 13. d = 50 added

**Problem.** The runtime exponent in d was fitted from 3 points, leaving almost no degrees of freedom for the interaction model.

**Fix.** `FEATURE_COUNTS = (50, 100, 300, 500)`. Four levels give 20 design points (Goldsmith et al., 2007; Montgomery, 2017). d = 50 is never underdetermined (every N ≥ 10·d), is the cheapest setting (an estimated ~6% extra runtime), and keeps nesting because subsets are prefixes of one ranking.

## 14. Same models and cases across variants

**Problem.** Within a run, every condition shares one set of fitted models and the same cases. But variants are separate runs that refit the models. Fixed seeds and single threading *should* make the fits identical, but nothing checked it.

**Fix.** Each run saves `model_fingerprint` (SHA-256 of the model's test-set probabilities, rounded to 10 decimals) and `explained_cases` in `run_meta.json`. `plot_results.check_same_models_across_variants` stops if any model's fingerprint or cases differ between variants. Tested: fingerprints matched across two smoke-test variants, and a deliberately altered fingerprint was caught.

**Why.** Paired comparisons are only valid when conditions share the same units and treatments (Montgomery, 2017, ch. 4).

## 15. Independent random streams per case

**Problem.** Repeat r always used seed 42 + r for every case, and the held-out set and references each used one seed for all cases. All cases therefore shared the same random-number streams, while the bootstrap and sign-flip tests assume cases are independent.

**Fix.** `config.case_seed(base, case_index)` derives a separate seed for each case using NumPy's `SeedSequence`. All ~23,500 possible seeds over the test set are distinct. Within a case, the same seed is shared by every model, d and N.

**Why.** Independence across units is what the tests assume. Sharing random numbers *within* a unit across the conditions being compared ("common random numbers") is a standard variance-reduction technique that keeps comparisons paired (Law, 2015).

## 16. Pinned-environment check

**Problem.** The smoke tests ran with six mismatches from `requirements-pinned.txt`: numpy, pandas, scipy, scikit-learn, matplotlib, and Python 3.11 instead of 3.13. Nothing detected this. Library versions change numerical results and speed (BLAS builds, tree inference, the LIME internals that `holdout_draw` mirrors), so runtime results would not be reproducible.

**Fix.** `protocol.check_versions` compares installed packages and the Python minor version with `requirements-pinned.txt` before any work. Full runs stop on any mismatch; quick runs warn. `--allow-version-mismatch` overrides this for non-paper runs. The result is saved in `run_meta.json`, and plotting warns about mismatched results.

## 17. Garbage collection and timing

**Problem.** `gc.collect()` before each timed call cost 42 ms × 9,000 calls, about 6 minutes per variant.

**Fix.** Automatic garbage collection is disabled *inside* each timed call (so a collection pause can never land in a measurement) and run every 100 calls outside timing. Garbage-collection pauses are a known source of timing noise (Georges et al., 2007).

## 18. Smaller bug fixes

- **Plot crash:** error bars failed when a confidence interval collapsed onto the mean (rounding produced a −1e-17 bar length). Error bars are now clipped at zero.
- **Speed probe:** the first probe round included BLAS warm-up (XGBoost's end/start ratio was 0.77). The first round is now discarded.
- **Threshold rounding:** see fix 3.
- **Pandas warning:** case selection added a column to the 561-column table on every run. It now samples on the labels alone. The selected rows are identical (verified).
- **Version guard on LIME internals:** the held-out sampler calls a private LIME method, so it refuses to run on any LIME version other than 0.2.0.1.

## Still open

- **Full run:** run the full study in the pinned environment on one machine, ideally the Mac used before (expected 3–4.5 hours for all three variants; an estimate). Keep the machine awake and close other heavy programs.
- **Correct-only analysis:** there is no version of the comparisons limited to rows every model predicts correctly.
- **Limitations to state in the paper:**
  - one dataset;
  - LIME samples features independently, producing increasingly unrealistic rows as d grows;
  - ranked subsets are redundant;
  - with 5 cases per class the bootstrap intervals are still somewhat too narrow;
  - R² *levels* across models reflect how linear each model is locally, so compare trends, not levels;
  - runtime results apply to these exact model settings, not model families.
- **Repository hygiene:** the 58 MB data archive, `__pycache__` files and smoke-test outputs were committed in `6e60dc9` and there is no `.gitignore`.

## References

- Bender, R., & Lange, S. (2001). Adjusting for multiple testing—when and how? *Journal of Clinical Epidemiology*, 54(4).
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Garreau, D., & von Luxburg, U. (2020a). Explaining the explainer: A first theoretical analysis of LIME. *AISTATS*.
- Garreau, D., & von Luxburg, U. (2020b). Looking deeper into tabular LIME. arXiv:2008.11092.
- Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically rigorous Java performance evaluation. *OOPSLA*.
- Goldsmith, S. F., Aiken, A. S., & Wilkerson, D. S. (2007). Measuring empirical computational complexity. *ESEC/FSE*.
- Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), ch. 7 and 18. Springer.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2).
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2).
- Kalibera, T., & Jones, R. (2013). Rigorous benchmarking in reasonable time. *ISMM*.
- Kvålseth, T. O. (1985). Cautionary note about R². *The American Statistician*, 39(4).
- Law, A. M. (2015). *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.
- Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley.
- Mytkowicz, T., Diwan, A., Hauswirth, M., & Sweeney, P. F. (2009). Producing wrong data without doing anything obviously wrong! *ASPLOS*.
- Nili, H., et al. (2014). A toolbox for representational similarity analysis. *PLoS Computational Biology*, 10(4).
- Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *JMLR*, 18.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Plumb, G., Molitor, D., & Talwalkar, A. (2018). Model agnostic supervised local explanations. *NeurIPS*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *KDD*.
- Slack, D., Hilgard, S., Singh, S., & Lakkaraju, H. (2021). Reliable post hoc explanations: Modeling uncertainty in explainability. *NeurIPS*.
- Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1).
- Visani, G., Bagli, E., Chesani, F., Poluzzi, A., & Capuzzo, D. (2022). Statistical stability indices for LIME. *Journal of the Operational Research Society*, 73(1).
- Zhang, Y., Song, K., Sun, Y., Tan, S., & Udell, M. (2019). "Why should you trust my explanation?" Understanding uncertainty in LIME explanations. arXiv:1904.12991.
