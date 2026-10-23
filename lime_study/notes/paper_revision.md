# Paper revision: review check, filled tables, cut-down text (23-10-2026)

All numbers below were re-read from `figures/combined_summary.csv`, `figures/paired_differences.csv`, `figures/reference_check.csv`, `figures/runtime_scaling.csv` and `results/*_performance.json` (full run, 21-10-2026, `claim-safe-fixes` design). Values are case-level means at N = 5,000 unless stated.

---

## 1. Are the reviewer's points correct?

| Review point | Verdict | Evidence |
|---|---|---|
| Abstract headline (ref. Jaccard 0.61–0.76 at d = 500) is not significant | **Correct** | d = 500: XGB vs RF p_Holm = 0.666, XGB vs LR p_Holm = 0.373. At d = 300 XGB beats both (p_Holm = 0.0036 each). Use d = 300 in the abstract. |
| V-D: LIME explains the probability, not the log-odds; LR is saturated near 0/1 | **Partly correct. Do not use it as the whole explanation.** | `make_predict_proba` confirms LIME sees probabilities. On the 30 cases, LR is saturated (p(true) > 0.98 or < 0.02 in 20/30; median 0.999). **But XGBoost is even more saturated (24/30), and it gives the *best* explanations at d ≥ 300.** RF is almost never saturated (2/30). So saturation explains the near-constant exclusions (LR 21, XGB 33, RF 0), but not why LR is worst. The LR-specific part is that its log-odds is a dense sum over all 561 features, so perturbing many features at once pushes the output through the steep part of the sigmoid in a step-like way. LR has the largest output spread (weighted SD 0.33–0.36 at d ≥ 300 vs 0.20–0.26 for XGB). Wording for this is in V-D below. |
| Cite S-LIME, Visani et al., Garreau & von Luxburg | **Correct, and all three exist** | Checked: Zhou, Hooker & Wang, KDD '21, pp. 2429–2438; Visani et al., JORS 73(1):91–101, 2022; Garreau & von Luxburg, AISTATS 2020, PMLR 108:1287–1296. |
| "which explains the near-constant outputs" should be "may explain" | **Correct** | Not tested. Saturation (above) is a second candidate cause. |
| Abstract says fidelity plateaus "by 5,000" but body says "2,500–5,000" | **Correct** | R² changes by < 0.005 between N = 5,000 and 10,000 in every cell, and by ≤ 0.015 from 2,500 to 10,000. Use "by N = 5,000" in both places. |
| Placeholders, `[verify]` tags, IEEE footer | **Correct** | Table I and Table II are filled in below. The 5 missing references still need to be added by you (see §4). |

### Further errors the review missed

1. **Wrong citation number.** III-G cites Holm as [14], but [14] is the LIME docs and Holm is [15].
2. **V-C is wrong for low d.** "Random forest produced lower RMSE than the other models" holds only for d ≥ 300. At d = 50 and 100, XGBoost has lower RMSE (0.014 and 0.053 vs 0.051 and 0.057). "Without having higher R²" is also false against LR, because RF's R² is higher than LR's at every d. The correct claim is that at d ≥ 300 RF had the lowest RMSE but lower R² than XGBoost.
3. **The Conclusion overclaims.** "XGBoost gave the highest mean agreement with the reference" is false at d = 50 and 100. RF is highest there (0.834 and 0.825 vs XGB 0.779 and 0.737). Restrict it to d ≥ 300.
4. **The stability-vs-d tests are not part of the 36-test family.** They are Holm-corrected in a separate table that covers all between-d comparisons (`analysis.py:346`). Say so. As written, a reader assumes they are part of the 36.

### Claims checked and correct
Runtime linear fits R² ≥ 0.993 (min 0.9933, RF d = 50); 8–10-fold from N = 500→10,000; b = 0.72–0.77 (1.65–1.70× per doubling); c = 0.66 / 0.83 / 0.90; 9–17 → 78–86 ms per 1,000 samples; gap 1.85× → 1.10×; < 10% runtime spread at d = 500 (0.438–0.480 s); 0.006 s vs 0.137 s per 0.01 Jaccard; N = d = 500 Jaccard 0.004–0.029; R² 0.08–0.30 at N ≥ 5,000; flat-case counts 21 / 33 / 0; reference ceilings 0.80–0.97; all stability-vs-d significance statements; LR lowest R² and lowest repeat Jaccard at every d.

---

## 2. Table I (filled in)

| Model | Library | Settings | Test acc. |
|---|---|---|---|
| LR | scikit-learn 1.7.2 | StandardScaler + LR; L2, C = 1, lbfgs, max_iter = 1000 | 95.5% |
| RF | scikit-learn 1.7.2 | 100 trees, default depth | 92.8% |
| XGB | XGBoost 3.2.0 | 250 trees, depth 6, learning rate 0.1, subsample 0.9 | 94.1% |

(Macro-F1 if wanted: 0.954 / 0.927 / 0.940.)

## 3. Table II (filled in, N = 5,000)

Bold = best per row. † = the d = 300 reference comparison where XGBoost is significantly better than both other models after Holm correction.

| Measure | d | LR | RF | XGB |
|---|---|---|---|---|
| Runtime (s) | 50 | **0.053** | 0.065 | 0.100 |
| Runtime (s) | 500 | **0.438** | 0.465 | 0.480 |
| Held-out R² | 300 | 0.085 | 0.198 | **0.254** |
| Held-out R² | 500 | 0.085 | 0.165 | **0.265** |
| Repeat Jaccard | 50 | 0.677 | **0.798** | 0.719 |
| Repeat Jaccard | 300 | 0.534 | 0.642 | **0.814** |
| Reference Jaccard | 50 | 0.729 | **0.834** | 0.779 |
| Reference Jaccard | 300 | 0.609 | 0.692 | **0.855**† |
| Reference Jaccard | 500 | 0.610 | 0.685 | **0.761** |

I added the d = 50 runtime and d = 50 reference rows so the table shows the narrowing runtime gap and RF's advantage at low d. That lets the prose drop those numbers.

---

## 4. Cut-down paper text

The body runs to about 2,050 words (was ~3,000), excluding tables, captions and references. Paste it section by section. `[CITE]` marks the references you still have to fill in.

### Abstract (~140 words)

Local Interpretable Model-agnostic Explanations (LIME) is widely used to explain black-box classifiers, yet there is little guidance on how its cost and explanation quality scale together as dimensionality grows. We ran a full factorial study on the UCI Human Activity Recognition dataset, crossing three models (logistic regression, random forest, XGBoost), four perturbed-feature counts (50–500) and five sample sizes (500–10,000), over 30 cases with five repeats each. Runtime grew almost linearly with sample size and sub-linearly with the number of perturbed features. Held-out surrogate fidelity levelled off by 5,000 samples and stayed low (R² ≤ 0.30), whereas agreement between repeated explanations kept improving up to 10,000 samples at a steeply rising marginal cost. Model choice mattered more for quality than for cost: at 500 features runtimes differed by under 10%, while at 300 features XGBoost's agreement with a 50,000-sample reference (0.86) significantly exceeded that of random forest (0.69) and logistic regression (0.61). Beyond the default budget, extra samples buy repeatability rather than faithfulness.

### I. Introduction

LIME [1] explains an individual prediction by perturbing the instance, querying the model on the perturbed samples and fitting a weighted linear surrogate whose largest coefficients form the explanation. Its runtime therefore depends on the number of samples N and on the model's inference cost, while explanation quality is expected to fall as the number of perturbed features d grows, because a fixed budget covers a larger space more thinly.

Prior work has studied LIME's instability, but little of it measures how cost and quality scale together as dimensionality and model complexity increase (Section II-B). This paper measures LIME's runtime against the stability and fidelity of its explanations for three models on a high-dimensional tabular dataset, to find where extra computation stops improving quality. It asks: (RQ1) how does runtime scale with N and d? (RQ2) how does explanation quality change with N and d? (RQ3) how does the black-box model change the cost/quality trade-off?

### II. Background and Related Work

**A. Related Work.** LIME's runtime has been reported only as a side result, not measured systematically, and not on high-dimensional tabular data [2]–[4]. More samples are known to improve stability at higher cost, but only for a single model [5]–[7]. S-LIME [16] goes further and chooses N with a hypothesis test so that the top features are stable, and Visani et al. [17] propose indices for measuring LIME's stability. Neither measures runtime, compares models or uses high-dimensional tabular data. Garreau and von Luxburg [18] show analytically that tabular LIME's explanations depend strongly on the kernel width, which bears on our design (Section VI). Finally, tabular LIME has been evaluated for usefulness without its cost being measured [1], [8]. This study times LIME systematically on high-dimensional tabular data, across several models, jointly with stability and fidelity.

**B. Choice of Models.** The three models represent widely used classifier families with different inference costs and decision surfaces. Logistic regression [9] is linear in the log-odds, so a linear surrogate should in principle suit it best. Random forest [10] averages bagged trees and gives a piecewise-constant output. XGBoost [11] boosts trees sequentially and gives sharper, more interaction-rich boundaries. Comparing them separates the model's effect on LIME's cost from its effect on explanation quality.

**C. Choice of Dataset.** UCI HAR [12] has 561 dense, continuous features from a waist-mounted smartphone's accelerometer and gyroscope, recorded from 30 participants performing six activities. This is high-dimensional for tabular data (though not p > n), and continuous features suit LIME's quartile discretisation. Its official split puts each participant in either the training or the test set, which avoids leakage; the original authors report about 96% accuracy on it [12].

> Comment CM1 ("why we chose it") is answered by "high-dimensional, dense, continuous, subject-disjoint split". Comment CM2 is answered for models by "linear vs bagged vs boosted". Both comments can be resolved.

### III. Methodology

**A. Study Design.** A full factorial experiment crosses three models, d ∈ {50, 100, 300, 500} and N ∈ {500, 1,000, 2,500, 5,000, 10,000}, with five repeats per cell, over 30 explained cases. The case is the unit of analysis, so repeats are averaged within a case first. Each case also receives two reference explanations at N = 50,000 per d.

**B. Data and Cases.** The official split is used: 21 participants (7,352 rows) for training and 9 (2,947 rows) for testing. Thirty test cases, five per class, were selected using true labels only, so no model is favoured, with at most four cases from any participant. Every model explains each case's true class.

**C. Models.** Table I lists the settings. Each model is trained once on all 561 features. Hyperparameters are not tuned, since the aim is to compare LIME across model families, not to maximise accuracy. Random state is 42 and inference is single-threaded.

[Table I]

**D. Perturbed Features and Sample Sizes.** The perturbed features are nested prefixes (50 ⊂ 100 ⊂ 300 ⊂ 500) of one seeded random permutation of the 561 features. LIME perturbs only these d features and freezes the rest at the explained row's values, so the model always receives all 561 inputs and stays fixed. The values of N bracket LIME's default of 5,000 (N:d from 1:1 to 200:1). The one cell with N ≤ d (N = d = 500) is excluded from trend lines and primary tests.

**E. LIME Configuration.** We use LimeTabularExplainer from LIME 0.2.0.1 [14]. Each of the d features is discretised into quartiles from 2,000 random training rows (about 500 per quartile). N samples are drawn by resampling from these bins, each is encoded by whether each feature stays in the instance's quartile, and each is weighted by an exponential kernel of the library-default width 0.75√d on its Euclidean distance from the instance. A ridge surrogate (α = 1) ranks all d features, and the top k = 10 are refitted (`highest_weights`). Seeds are derived per case with NumPy's SeedSequence, so every random choice is reproducible.

**F. Metrics.** *Cost* is the wall-clock time of each `explain_instance()` call (perf_counter). Conceptually,

T(N, d) ≈ T_sample + T_predict + T_weight + T_fit,  (1)

where sampling and weighting scale as O(N·d), and T_predict (N model queries) is the only model-dependent term. *Fidelity* is kernel-weighted R² and RMSE of the surrogate against the model on 5,000 held-out perturbations per case and d, shared across N, repeats and models. R² is excluded where the model's output is near-constant (weighted SD < 0.01): this applies to 21 of 120 case × d combinations for LR, 33 for XGB and none for RF. *Stability* is the mean top-5 Jaccard overlap over the ten pairs of repeats. *Convergence* is top-5 Jaccard against the two N = 50,000 references, which were accepted only if they agreed with each other at ≥ 4/6 on average. All 12 model × d combinations passed (0.80–0.97).

**G. Protocol and Analysis.** Models run interleaved in randomised blocks, one per (case, d, N, repeat), so machine drift cannot favour a model. All runs were single-threaded, in one process, on an 8-core Apple Silicon Mac (Python 3.13.7), with untimed warm-ups. Uncertainty is estimated by a class-stratified case bootstrap (2,000 resamples, 95% intervals), with participant-clustered intervals as a check. Model differences are tested with paired sign-flip permutation tests (10,000 flips), Holm-corrected [15] within a pre-specified family of 36 tests: three model pairs × four d at N = 5,000, for runtime, RMSE and reference Jaccard. Comparisons between values of d are Holm-corrected as a separate family. Runtime scaling is summarised per model by

log t = a + b log N + c log d.  (2)

### IV. Results

Fig. 1 shows runtime and quality against N; Table II gives the main values at N = 5,000.

[Fig. 1]

**A. Cost (RQ1).** Runtime grew almost linearly with N on top of a fixed overhead. In (2), b ≈ 0.72–0.77, so each doubling of N multiplied runtime by 1.65–1.70, below 2 because of the overhead; c ≈ 0.66 for XGBoost, 0.83 for random forest and 0.90 for logistic regression. The cost per 1,000 samples rose from 9–17 ms at d = 50 to 78–86 ms at d = 500. Logistic regression was fastest and XGBoost slowest in every cell, but the gap narrowed from 1.85× at d = 50 to 1.10× at d = 500 (N = 10,000).

**B. Explanation Quality (RQ2).** Held-out R² changed by less than 0.005 between N = 5,000 and 10,000 in every cell and stayed low (0.08–0.30), so extra samples did not make the surrogate more faithful. Agreement between explanations kept improving up to N = 10,000 at a rising cost (Fig. 2). For XGBoost at d = 500, each 0.01 gain in repeat Jaccard cost 0.006 s from N = 1,000 to 2,500, but 0.137 s from 5,000 to 10,000. At N = d = 500, repeat agreement was near zero (0.004–0.029). Stability fell significantly with d for random forest (d = 50 and 100 each vs 300 and 500) and logistic regression (d = 50 vs 300), but not for XGBoost.

[Fig. 2]

**C. Model Effects (RQ3).** Model choice mattered more for quality than for cost. At d = 500, runtimes differed by under 10%, while the models differed in quality (Table II). At d = 300, XGBoost agreed with the reference significantly more than both other models (p_Holm < 0.01); at d = 500 its lead was not significant (p_Holm ≥ 0.37). At low d the ranking reversed: random forest was the most stable and the closest to the reference at d = 50. Logistic regression, the most accurate model, had the lowest R² at every d and the lowest stability at N = 5,000.

[Table II]

### V. Discussion

**A. Why the Runtime Gap Narrows.** LIME's sampling and weighting scale with N·d, but inference does not depend on d because the model always receives all 561 inputs. At high d, LIME's own overhead therefore plausibly dominates, shrinking the gap between models. XGBoost's lower d-exponent fits this: its larger, d-independent inference cost is a bigger share of its total. Because stages were not timed separately, this remains to be confirmed.

**B. Fidelity Plateau versus Stability Gains.** Fidelity is limited by how well a linear model on binary quartile indicators can approximate the model locally. Once the coefficients are estimated well enough, more samples cannot remove this structural error. Stability depends on the variance of the coefficients, which keeps shrinking with N. Beyond the default, extra samples buy repeatability, not faithfulness. At N = d = 500 the surrogate is barely determined and its rankings are close to random.

**C. R² versus RMSE.** At d ≥ 300, random forest had the lowest RMSE but lower R² than XGBoost. Its output barely moves within the neighbourhood (weighted SD ≈ 0.06 at every d, against 0.20–0.36 for the others at d ≥ 300), so any surrogate achieves a small absolute error. RMSE is therefore not comparable across models whose outputs vary by different amounts; R² normalises by that variation and is the fairer comparison.

**D. The Most Accurate Model Gave the Weakest Explanations.** Logistic regression is linear in the log-odds, but LIME explains its predicted probability, and quartile binarisation removes the linearity in any case. On 20 of our 30 cases the model was saturated (p > 0.98 or p < 0.02). Because its log-odds sums contributions from all 561 features, perturbing many features at once moves the output through the steep part of the sigmoid in a step-like way. Its output had the largest spread of the three models (weighted SD 0.33–0.36 at d ≥ 300), which a linear surrogate on binary features fits poorly. Saturation alone does not explain the result, since XGBoost was saturated on even more cases (24 of 30). The dense, all-feature dependence of logistic regression is the plausible difference, but it was not tested. Explaining the log-odds instead of the probability would test it directly.

### VI. Limitations

The default kernel width 0.75√d changes with d, and LIME's explanations are known to depend strongly on this width [18], so part of the apparent effect of d may be the kernel's. The perturbed features are random, not the most important, which may explain the near-constant outputs at low d, alongside output saturation. Fixed-kernel and ranked-subset variants exist in the code but were only smoke-tested. Absolute runtimes are specific to one single-threaded machine; scaling exponents and relative comparisons should transfer. RMSE is not comparable across models, and R² covers different cases per model. Only 30 cases from 9 participants on one dataset were explained, and participant-clustered intervals are wider. LIME samples features independently, so it queries the model on sensor combinations that cannot occur. Each model was run in a single untuned configuration, so the findings describe these models rather than their families.

### VII. Conclusion

Runtime grows almost linearly with N and sub-linearly with d (RQ1). Held-out fidelity levels off by N = 5,000 and stays low, while stability keeps improving at a steeply rising marginal cost, so beyond LIME's default, extra samples buy repeatability, not faithfulness (RQ2). The model affects quality far more than cost: XGBoost gave the most convergent explanations at d ≥ 300, random forest at low d, and logistic regression, the most accurate model, the weakest throughout (RQ3). In practice, for d ≥ 300, N = 5,000 is a reasonable floor and N ≤ d should be avoided. Users who need faithful explanations should measure fidelity directly rather than raise the budget. Future work should run the fixed-kernel and ranked-subset variants in full, explain log-odds rather than probabilities, and extend the study to more datasets and models.

---

## 5. New references (checked 23-10-2026)

[16] Z. Zhou, G. Hooker, and F. Wang, "S-LIME: Stabilized-LIME for model explanation," in *Proc. 27th ACM SIGKDD Conf. Knowledge Discovery and Data Mining*, 2021, pp. 2429–2438.
[17] G. Visani, E. Bagli, F. Chesani, A. Poluzzi, and D. Capuzzo, "Statistical stability indices for LIME: Obtaining reliable explanations for machine learning models," *J. Oper. Res. Soc.*, vol. 73, no. 1, pp. 91–101, 2022.
[18] D. Garreau and U. von Luxburg, "Explaining the explainer: A first theoretical analysis of LIME," in *Proc. 23rd Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 108, 2020, pp. 1287–1296.

Removed with the cuts: [13] Covertype (sentence deleted). If you keep the IEEE ordering-by-first-citation rule, renumber after deleting [13].

Still to do yourself: full entries for [2] Ross, [3] Mougan Navarro, [4] Bladen, [6] Burger, [7] Knab; the remaining `[verify]` tags; the [14] access date; deleting the `XXX-X-XXXX…©20XX IEEE` footer; and checking the course's assigned readings for anything that belongs in II-A.
