# Explainable AI: LIME Cost/Quality Scaling on High-Dimensional Tabular Data across Multiple Different Models

**By** Caitlin Amelia Moloney (24411086), David Moore (24419095), Hugh Murtagh (24421448), Sebastian Kimmel (24420336)

---

## 0. Abstract

---

## 1. Introduction

LIME is a popularly used model-agnostic explainability method for black box models. The runtime of LIME depends on the number of perturbation samples and inference cost of the black-box model to which it is being applied, while the quality of explanations degrades as feature space grows.

Studies have examined LIME's instability and proposed variations upon LIME, but there is a lack of guidance on how cost and quality might scale together as dimensionality and model complexity increase.

This paper's purpose is to investigate LIME cost/quality scaling across multiple models on large tabular datasets, measuring runtime against stability and fidelity to identify where additional computations stop improving explanation quality.

---

## 2. Background/Related Work

### 2.0 Research Question

Given this project mixes external research and internal experimentation, the research and preparation pursued in this project required thorough attention. This began from the initial steps of this project, the ideation of a topic worth discussing. We were drawn to explanation methods as a topic due to our lecture on SHAP at the time, after multiple renditions of ideas related to explanation methods including a comparison between SHAP and LIME, we were drawn to just highlighting LIME given our projects scope, our timeline, and general interest into an explanation method we had not yet covered during lectures.

> **Comment [UL1]:** Repeat of "we were drawn to".

LIME also provided a broad research field due to its wide use as well as its ability to work with any model. After the topic of LIME was chosen, we then had to settle a specific question which came to by combining 3 initial simple ideas.

1. How does runtime grow with samples (N) and features (d)?
2. How does explanation quality change with N and d? and
3. Does the model type change the cost/quality trade-off?

We settled these three questions during our research into current LIME related papers. From these paper's we spotted three key themes shared across a selection of them;

- **Theme 1:** LIME's runtime has only been timed as a side result, never systematically on high-dimensional tabular data (Ross et al.; Mougan Navarro; Bladen et al.).
- **Theme 2:** More samples improve stability but cost more, but this trade-off is only described or shown for one model (Zhao et al.; Burger et al.; Knab et al.).
- **Theme 3:** Tabular LIME has been evaluated for usefulness, but its cost was never measured (Dieber & Kirrane; Ribeiro et al.).

This resulted in our final topic choice: an investigation into LIME's Cost/Quality Scaling on High-Dimensional Tabular Data across multiple different models.

### 2.1 Dataset

> **Comment [UL2]:** This section needs references - where did the info come from?

The dataset we chose for our experiment on LIME was the UCI Human Activity Recognition (HAR) database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. This dataset was chosen to allow for a perspective of LIME and dimensionality, and UCI HAR's 561 features make it high-dimensional compared with typical tabular datasets, though not in the strict statistical sense of having more features than rows. UCI HAR is pre-randomly partitioned for a 70% volunteer training set (21 people, 7,352 rows), and a 30% testing set (9 people, 2,947 rows). We proceeded with UCI HAR's split since the original HAR paper resulted to be 96% accurate, as well as that, no person is in both sets to prevent leakage. Finally, a random state seed of 42 was used so results can be reproduces exactly since every model can get the same "random" choices.

### 2.2 Models

> **Comment [UL3]:** No need to mention tuning this is more applicable to methodology and covered there - discuss moreso why each model was chosen - use examples from powerpoint presentation. and provide sources in this section to show where we did our research.

The choice of using multiple models benefits our experiment by serving a range of methods used by artificial intelligence; linear to tree ensemble to boosted trees. The three models used during the experiment are logistic regression, random forest, and XGBoost. Although sensible settings were applied to each model (LogReg: scaled, 1,000 iterations. RF: 100 trees. XGBoost: 250 trees, depth 6, learning rate 0.1), no tuning was done as we want a fair study of a standard LIME, so models only need to be reasonable

### 2.3 Experiment Design

> **Comment [UL4]:** Need to merge this section with Methodology - a lot of repetition but deeper explanations. We can address claude to see what stays in background vs methodology.

Careful consideration was needed to ensure a fair experiment between each model. 30 of the 2,947 test rows were explained, 5 per activity, to provide an efficient LIME use while using enough rows to not allow the results to become too narrow.

Before the model is involved, the rows are picked from true labels from the dataset only, rather than predicted labels. This is to avoid the scenario if some rows suit certain models more than others. Given the nature of the dataset discussed in section 2.2, there is a maximum of 4 rows covering a single person as rows from one person alone would be too similar and would dominate.

All three models answer the same question about the same row by using the true class, so their explanations can be compared fairly. While the models of course need training to make their predictions, 2,000 random training rows are also given to LIME so that it learns realistic feature values from them.

Since LIME splits each feature's values into four ranges. 2,000 rows give about 500 rows per quarter, which we found enough to be accurate, and quick. Ten features are reported, as described in LIME's official document.

When choosing sample sizes (N) we decided on a rough spread around LIME's default sample size of 5,000 (500, 1,000, 2,500, 5,000, 10,000). We found that this gave us an accurate way to trace the cost/quality curve.

In terms of feature sizes (d), we settled on a range of 100, 300, and 500 which allows us to respect a study using hundreds of features in equal steps. We also included a low baseline of just 50 features to contrast the high 500/561 feature size.

Since the set random seed of 42 is used to shuffle all 561 features, subsets are built by taking the first d features of the feature size. This means that a bigger d only adds more features to the one before, making the subsets nested, stopping random feature picks.

We chose that features that LIME does not use from the feature size become frozen to ensure the model stays the same, only the LIME change varies.

To examine whether LIME gives the same answer twice, repeating LIME explanations shows how stable LIME is. We chose 5 runs so that there were 10 pairs to compare, a balance between information and time.

Lastly, for the fair experiment design discussion, we ran the experiment models in one process in a mixed random order so that if a slowdown were to occur, it would affect every model equally. Another bonus to running a single process is that each model shared the same setup of the laptop the experiment was run on.

---

## 3. Methodology

### 3.0 Study Design

The full factorial experiment detailed in this section was designed to characterise how LIME cost and quality scale as dimensionality grows on a tabular dataset. It crosses three models, four perturbed-feature counts d ∈ {50, 100, 300, 500} and five sample sizes N ∈ {500, 1,000, 2,500, 5,000, 10,000}, with five repeats per cell, over 30 explained cases.

The explained case is the unit of analysis; repeats are averaged within a case first ns at N = 50,000 (240 per model)

The design of this experiment aims to address three questions:

1. How does runtime scale with N and d?
2. How does explanation quality change with N and d?
3. How does the black-box model change the cost/quality trade-off?

### 3.1 Dataset & Explained Cases

The UCI Human Activity Recognition (HAR) dataset has 561 continuous features and six activity classes. It is used with its official subject-based split: 21 participants (7,352 rows) for training and 9 participants (2,947 rows) for testing. Thirty test cases, five per class, were selected to balance classes and participants without reference to any model's predictions. Every model explains each case's true class.

### 3.2 Models

The following table details each model and its settings for this experiment:

**Table 1. Model Settings**

| Model | Library | Settings |
|---|---|---|
| Logistic Regression (LR) | Scikit-learn 1.7.2 | StandardScaler + LR pipeline; L2, C = 1, lbfgs |
| Random Forest (RF) | Scikit-learn 1.7.2 | 100 trees |
| XGBoost (XGB) | Xgboost 3.2.0 | 250 trees, depth 6, learning rate 0.1, subsample 0.9 |

Each model is trained once on all 561 features and is not retrained for each value of d. Hyperparameters are neither tuned nor matched across models, since the aim is to compare LIME's behaviour on representative model families, not to maximise accuracy. The random state is 42, and all inferences were single-threaded.

### 3.3 Features & Sample-Size Axes

**Perturbed Features (d):** d ∈ {50, 100, 300, 500}. These are nested prefixes (50 ⊂ 100 ⊂ 300 ⊂ 500) of one seeded random permutation of the 561 features. LIME perturbs only these d features. The rest stay at the explained row's values, so models always receive all 561. d = 50 serves as a low-dimensional baseline.

**Samples (N)** ∈ {500, 1,000, 2,500, 5,000, 10,000}. This range brackets LIME's default of 5,000, giving N:d ratios from 1:1 to 200:1. The single cell with N ≤ d (N = d = 500) is flagged and excluded from trend lines and primary tests.

### 3.3 LIME Configuration

Explanations are generated with `LIMETabularExplainer` from LIME 0.2.0.1. The following are the LIME configuration settings:

- Quartile discretisation
- Euclidean distance
- Exponential kernel with the library-default width of 0.75√d (the explainer sees only the d perturbed columns)
- Ridge surrogate with α = 1
- 2,000 training rows as background data
- k = 10 reported features, so `feature_selection="auto"` resolves to `highest_weights`

Each (case, d, N) cell is repeated 5 times. Explainer seeds are derived per case using NumPy's `SeedSequence`.

### 3.4 Evaluation Metrics

#### 3.4(a) The cost of LIME

Cost is the wall-clock time of each `explain_instance()` call, measured with `perf_counter`. Stages were not timed separately, but conceptually:

$$T(N,d) \approx T_{sample}(N,d) + T_{predict}(N, \text{model}) + T_{weight}(N,d) + T_{fit}(N,d,k) \quad (1)$$

- T_sample and T_weight (perturbation sampling, distance and kernel computation) each scale roughly as O(N·d).
- T_predict covers the N black-box predictions. It is the only term that depends on the model.
- T_fit is one ridge fit over all d features to rank them, followed by a refit on the k = 10 selected features.

#### 3.4(a) The quality of LIME

In each (model, d, N) cell, runtime is paired with three quality measures.

1. **Local fidelity:** The surrogate is scored against the model on 5,000 held-out perturbations per case and d, using kernel-weighted R² and RMSE. The held-out set is shared across all N, repeats and models, so comparisons are paired. R² is excluded where model output is near-constant (weighted SD < 0.01). This affects 21/120 LR and 33/120 XGB case × d combinations, and none for RF, so R² means cover different case sets per model. LIME's in-sample score is reported for comparison only.
2. **Stability:** the mean Jaccard overlap of top-5 features across the 10 pairwise comparisons of the 5 repeats.
3. **Convergence:** the top-5 Jaccard overlap against the two N = 50,000 reference explanations. References for a model × d combination are accepted only if their mean agreement across cases is at least 4/6, which corresponds to sharing 4 of the top 5 features. All 12 combinations passed (0.80–0.97), so no comparison was dropped.

### 3.5 Measurement Protocol & Analysis

Models run interleaved in randomised blocks, one block per (case, d, N, repeat), with model order shuffled within each block, so machine drift cannot favour one model. BLAS and OpenMP are single-threaded; warm-up runs are untimed, and explainer construction and scoring fall outside the timed region. Experiments ran on Python 3.13.7 on an Apple Silicon Mac (8 cores) with global seed 42.

Uncertainty is estimated with a class-stratified case bootstrap (2,000 resamples, 95% percentile intervals). Participant-clustered intervals serve as a conservative check.

Model differences are tested with paired sign-flip permutation tests (10,000 random flips). Holm correction is applied within a pre-specified family of 36 tests: the model pairs at each d at N = 5,000, for runtime, RMSE and reference Jaccard.

Runtime scaling per model is summarised by:

$$\log t = a + b \log N + c \log d \quad (2)$$

---

## 4. Results and Analysis

LIME's runtime grew almost linearly with the number of perturbation samples N, with a fixed per-explanation overhead (straight-line fits, R² ≥ 0.993). Increasing N from 500 to 10,000 raised runtime ~8-10 fold.

The cost of each sample grew with the number of perturbed features, from 9–17 ms per 1,000 samples at d = 50 to 78–86 ms at d = 500. Logistic Regression was fastest and XGBoost slowest in every setting. However, the gap narrowed from 1.85× at d = 50 to 1.10× at d = 500 (N = 10,000), plausibly because LIME's own sampling and weighting, which scale with N·d, outweigh model inference at high d; individual stages were not timed.

Held-out fidelity changed little beyond 2,500–5,000 samples and remained low (R² = 0.08–0.30), so within the tested range additional samples did not make the surrogate more faithful. R² excludes cases whose output barely varied.

Explanation agreement, by contrast, kept improving up to 10,000 samples at a rising cost. For XGBoost at d = 500, each 0.01 gain in repeat Jaccard cost 0.006 s between 1,000 and 2,500 samples, but 0.137 s between 5,000 and 10,000. At the one setting where N equalled d (N = d = 500), repeat agreement was very low (Jaccard 0.004–0.029).

Model choice mattered more for quality than for cost. At N = 5,000 and d = 500, runtimes differed by under 10% across models, whereas agreement with the 50,000-sample reference ranged from 0.61 to 0.76. Random Forest was the most stable model at d = 50 (0.80) and XGBoost at d = 300 (0.81). At d = 300, XGBoost's agreement with the reference (0.855) exceeded that of Random Forest (0.692) and Logistic Regression (0.609), and these differences survive Holm correction.

Stability declined significantly with d for Random Forest and for Logistic Regression (d = 50 vs 300); no significant decline was detected for XGBoost. Logistic Regression, despite the highest test accuracy (95.5%), produced the least faithful explanations at every d and the least stable at N = 5,000.

---

## 5. Conclusion & Limitations

### 5.1 Conclusion

Throughout this study, we examined three main themes: computational cost, explanation quality, and model-dependent trade-offs, addressed through three research questions. Firstly, runtime increased with both perturbation sample size (N) and feature dimensionality (d), with Logistic Regression consistently the fastest, followed by Random Forest and XGBoost.

Secondly, increasing N generally improved explanation fidelity and stability, although these improvements diminished at higher sample counts, while increasing dimensionality often reduced explanation consistency.

Thirdly, the underlying model significantly influenced the cost-quality trade-off. Our experiments showed that XGBoost achieved the highest mean held-out fidelity (R²) and explanation stability at 300–500 features, while Random Forest produced lower absolute surrogate error (RMSE) at these dimensions. Logistic Regression offered the lowest computational cost but generally weaker explanation quality.

Overall, XGBoost demonstrated the strongest quality-focused performance in our higher-dimensional experiments, although no single model consistently outperformed the others across all metrics.

### 5.2 Limitations

Several limitations affected the scope of this study. TabFM was excluded due to its computational demands, with predictions alone estimated to require approximately ~34 hours.

The original Covertype dataset was replaced since its 54 features were insufficient for our intended high-dimensional analysis.

Spearman correlation was removed because its model-dependent chance levels complicated stability comparisons, leaving top-five Jaccard as our repeatability measure.

Furthermore, the study was restricted to one dataset, UCI HAR, and 30 explained instances. The dataset's highly correlated sensor features may also produce unrealistic LIME perturbations. These factors limit the generalisability of our findings to other datasets, models, and experimental configurations.

---

## 6. References

- Lime github
- https://lime-ml.readthedocs.io/en/latest/lime.html
