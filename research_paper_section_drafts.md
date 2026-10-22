 
# Methodology
## 3.0 Study Design 

This controlled computational experiment measures how LIME’s cost and quality scale with the number of perturbation samples (N) and perturbed features (d). The full factorial design crosses 3 models × 4 values of d × 5 values of N × 5 repeats over 30 explained cases. This gives: 

- 9,000 timed explanations (3,000 per model) 
- 18,000 repeat pairs for measuring stability (6,000 per model) 
- 720 reference explanations at N = 50,000 (240 per model) 

The design addresses three questions: 

1. How does runtime scale with N and d? 
2. How does explanation quality change with N and d? 
3. How does the black-box model change the cost/quality trade-off? 

The explained case is the unit of analysis. Repeats are averaged within a cse first. 

## 3.1 Dataset & Explained Cases 

The UCI Human Activity Recognition (HAR) dataset has 561 continuous features and 6 activity classes. It was chosen for its high dimensionality and continuous features. The official subject-based split is used: 21 participants (7,352 rows) for training and 9 participants (2,947 rows) for testing, so no participant appears in both.  Five test cases are drawn per true class, giving 30 cases. Selection runs round-robin over classes. Each pick comes from the least-used participant, and no participant contributes more than four cases. Cases are selected without reference to any model’s predictions. Every model explains the case’s true class. Means are therefore taken over a class-balanced set, not the test-set class distribution. 

## 3.2 Models 

The following table details each model and its settings for this experiment: 

## Table 1. Model Settings 

Model 

Library 

Settings 

Logistic Regression (LR) 

Scikit-learn 1.7.2 

StandardScaler + LR pipeline; L2, C =1, lbfgs 

Random Forest (RF) 

Scikit-learn 1.7.2 

100 trees 

XGBoost (XGB) 

Xgboost 3.2.0 

250 trees, depth 6, learning rate 0.1, subsample 0.9 

 

Each model is trained once on all 561 features and is not retrained for each value of d. Hyperparameters are neither tuned nor matched across models, since the aim is to compare LIME’s behaviour on representative model families, not to maximise accuracy. The random state is 42, and all inference was single-threaded. Test accuracies are 95.5% (LR), 92.8% (RF) and 94.1% (XGB). Of the 30 cases, LR predicts 27 correctly; RF and XGB predict 28 each. 

## 3.3 Features & Sample-Size Axes 

Perturbed Features (d): d ∈ {50, 100, 300, 500}. These are nested prefixes (50 ⊂ 100 ⊂ 300 ⊂ 500) of one seeded random permutation of the 561 features. LIME perturbs only these d features. The rest stay at the explained row’s values, so models always receive all 561. d = 50 serves as a low-dimensional baseline. 
Samples (N) ∈ {500, 1,000, 2,500, 5,000, 10,000}. This range brackets LIME’s default of 5,000, giving N:d ratios from 1:1 to 200:1. The single cell with N ≤ d (N = d = 500) is flagged and excluded from trend lines and primary tests. 

## 3.3 LIME Configuration 

Explanations are generated with LIMETabularExplainer from LIME 0.2.0.1. The following are the LIME configuration settings: 

- Quartile discretisation 
- Euclidean distance 
- Exponential kernel with the library-default width of 0.75√d (the explainer sees only the d perturbed columns) 
- Ridge surrogate with α = 1 
- 2,000 training rows as background data 
- k = 10 reported features, so feature_selection="auto" resolves to highest_weights 

Each (case, d, N) cell is repeated 5 times. Explainer seeds are derived per case using NumPy’s SeedSequence since the default kernel width grows with √d, part of any effect of d may reflect the wider kernel (Section V-B). 

## 3.4 Evaluation Metrics 

### 3.4(a) The cost of LIME 

Cost is the wall-clock time of each explain_instance() call, measured with perf_counter. Stages were not timed separately, but conceptually: 

- T(N,d) ≈ T_sample(N,d) + T_predict(N, model) + T_weight(N,d) + T_fit(N,d,k) (1) 

- T_sample and T_weight (perturbation sampling, distance and kernel computation) each scale roughly as O(N·d). 

- T_predict covers the N black-box predictions. It is the only term that depends on the model. 

- T_fit is one ridge fit over all d features to rank them, followed by a refit on the k = 10 selected features. 

### 3.4(a) The quality of LIME 

In each (model, d, N) cell, runtime is paired with three quality measures. 

**Local fidelity: **The surrogate is scored against the model on 5,000 held-out perturbations per case and d, using kernel-weighted R² and RMSE. The held-out set is shared across all N, repeats and models, so comparisons are paired. R² is excluded where model output is near-constant (weighted SD < 0.01). This affects 21/120 LR and 33/120 XGB case × d combinations, and none for RF, so R² means cover different case sets per model. LIME’s in-sample score is reported for comparison only. 
**Stability:** the mean Jaccard overlap of top-5 features across the 10 pairwise comparisons of the 5 repeats. 
**Convergence:** the top-5 Jaccard overlap against the two N = 50,000 reference explanations. References for a model × d combination are accepted only if their mean agreement across cases is at least 4/6, which corresponds to sharing 4 of the top 5 features. All 12 combinations passed (0.80–0.97), so no comparison was dropped. 

## 3.5 Measurement Protocol 

All three models run interleaved in one process as a randomised complete block design. A block is one (case, d, N, repeat) combination. Model order is shuffled within each block so machine drift cannot systematically favour one model. 
Further timing controls: 

- BLAS and OpenMP are limited to one thread. 
- Untimed warm-up explanations run for each (model, d) pair. 
- Garbage collection is disabled inside timed calls. 
- Explainer construction and quality scoring are excluded from the timed region. 
- Eleven speed probes spread across the run monitor drift. 

## 3.6 Statistical Analysis 

Uncertainty is estimated with a class-stratified case bootstrap (2,000 resamples, 95% percentile intervals). Participant-clustered intervals are computed as a conservative check. Model differences are tested with paired sign-flip permutation tests. Enumerating all 2³⁰ sign flips is infeasible, so 10,000 random flips are used. Holm correction is applied within a pre-specified primary family of 36 tests: three model pairs × four values of d at N = 5,000, for each of runtime, held-out RMSE and reference Jaccard. Runtime scaling per model is summarised by: 

log t = a + b log N + c log d (2) 

where b and c are the scaling exponents in N and d. 

## 3.7 Controls & Reproducibility 

The software environment is pinned to Python 3.13.7, NumPy 2.2.6, scikit-learn 1.7.2, XGBoost 3.2.0, LIME 0.2.0.1, pandas 2.3.3, SciPy 1.15.3 and matplotlib 3.10.9. A full run refuses to start if any installed version differs. The global seed is 42. Experiments ran on an Apple Silicon Mac (arm64, 8 cores), and the full benchmark took 4,123 s (≈69 min). Speed probes ranged from −0.3% to +10.9% of the first probe. The +10.9% was a single probe; the other ten stayed within about ±4%. Each run records model fingerprints and a run ID. 
