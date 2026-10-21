> **Superseded (20-10-2026):** this note describes the earlier Covertype setup. The study now uses UCI HAR; see `lime_study_rebuild_plan.md` and `dataset_justification.md`.

# Experiment handoff: LIME cost and quality on Covertype

## Purpose

Prepare and run a controlled comparison of LIME explanations for three seven-class classifiers on the Forest CoverType dataset:

- XGBoost
- Logistic Regression
- Random Forest

The main comparison is how LIME runtime, local fidelity, and explanation stability vary with **18, 36, and 54 perturbed features**. This is a LIME explanation-dimension sweep: every classifier is trained on the same full 54-feature input. The models are not retrained using 18- or 36-feature inputs.

Do not treat the notebook's older one-row/sample-budget experiments as the main study. The reusable protocol is `experiments/controlled_lime_protocol.py`.

## Canonical protocol already in the repository

`experiments/controlled_lime_protocol.py` is the shared source of truth for the dataset split, explanation cases, feature subsets, and LIME measurements. Read it before changing any experiment notebook.

### Dataset and split

- Load with `sklearn.datasets.fetch_covtype(as_frame=True)`; the full dataset has 581,012 rows and 54 predictors.
- The 54 columns are the first 10 numeric terrain/distance/hillshade variables and 44 binary indicators (4 wilderness areas and 40 soil types).
- Target labels are shifted from 1–7 to 0–6 in `load_controlled_covtype()` to match scikit-learn model classes. `CLASS_NAMES` maps those positions to the seven cover types.
- Take the same stratified sample of 10,000 rows (`random_state=42`), then split it into 8,000 training and 2,000 test rows with a stratified 80/20 split (`random_state=42`).
- All model runs must use this same sampled data and split. The test cases are drawn from the test set, three rows per true class, selected with seed 42. Case selection does not depend on any model's prediction.

### Feature-count conditions

`feature_subset()` always includes the first 10 numeric features, then adds a seeded, shared permutation of binary feature indices to make nested subsets of 18, 36, and 54 columns. Thus the 18-column subset is contained in the 36-column subset, which is contained in the full 54-column subset.

For each explanation, features outside the selected subset are held at that test case's observed values. LIME perturbs only the selected columns. The model still predicts on complete 54-column rows. This isolates explanation perturbation dimension; it does not test the predictive impact of feature selection or dimensionality reduction.

### Cases, LIME settings, and run count

- Background data: 2,000 training rows sampled with seed 42.
- Cases: 3 held-out rows for each of 7 classes (21 cases total).
- Feature counts: 18, 36, 54.
- Perturbation budgets (`num_samples`): 500, 1,000, 2,500, 5,000, 10,000.
- Repeats: 5 per case, feature count, and budget. Seeds are 42 through 46, with a new explainer constructed for each repeat.
- Number of displayed LIME features: 10.
- Top-feature stability cutoff: 5.
- Runtime uses `time.perf_counter()` around `explain_instance()`. The first prediction call is warmed up before timings. Timing includes LIME perturbation generation, model probability calls, and surrogate fitting; it excludes model fitting, the warm-up, data preparation, and explainer construction.

This is **21 × 3 × 5 × 5 = 1,575 LIME explanations per model** (4,725 across the three models). Each model also produces 3,150 pairwise stability comparisons (21 cases × 3 dimensions × 5 budgets × 10 run pairs).

### Measurements and interpretation

- **Runtime:** raw seconds per explanation; summary reports mean, median, and standard deviation.
- **Fidelity:** `explanation.score`, LIME's local surrogate fit score (R²-style). It is not test accuracy and should be labelled local fidelity/surrogate fit.
- **Stability:** compare repeat explanations at the same case, feature count, and sample budget using pairwise Spearman correlation and top-5 Jaccard overlap; summarize by mean. Zero weights are used for unreported features. Spearman is computed over the union of features selected in either explanation, so shared unselected zeros do not dominate it.
- Report held-out predictive performance separately from LIME diagnostics. The helper returns accuracy, balanced accuracy, macro precision, macro recall, and macro F1 on the full 2,000-row test split.
- Summaries are grouped by model × feature count × perturbation budget. Preserve the raw rows so aggregation and plots can be audited.

## Model definitions in the experiment notebooks

Keep these existing configurations unless the research team explicitly agrees to a change. Record library versions and any configuration changes alongside the outputs.

| Model | Current notebook configuration | Notes |
|---|---|---|
| XGBoost | `XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=4)` | See `experiments/coverType_XGBoost_activity.ipynb`. Use `predict_proba` through a DataFrame wrapper with the original feature names. |
| Logistic Regression | `Pipeline(ColumnTransformer([("numeric", StandardScaler(), first 10 columns)], remainder="passthrough"), LogisticRegression(max_iter=1000, solver="lbfgs"))` | See `experiments/CovTypeLogRegLimeExp.ipynb`. The numeric features are standardized; binary indicators remain unchanged. Use the pipeline's `predict_proba` through a DataFrame wrapper. |
| Random Forest | `RandomForestClassifier(n_estimators=100, random_state=42)` | See `experiments/randomForest.ipynb`. Use `predict_proba` through a DataFrame wrapper. |

### What is and is not standardized across models

The protocol standardizes the data sample and split, selected LIME cases, explanation settings, and evaluation measures. It does **not** standardize model hyperparameters or tune the classifiers. Each model is fitted with the configuration listed above; comparisons therefore use a shared evaluation setup, not a shared or equally optimized model configuration.

The most relevant overlap in model settings is:

- **`random_state`**: explicitly set to 42 for Random Forest and XGBoost. Logistic Regression does not set it in this notebook. It can accept a `random_state`, though its effect depends on the solver; the current `lbfgs` setup is deterministic.
- **`n_estimators`**: used by Random Forest and XGBoost to set the number of trees (100 and 250, respectively). It does not apply to Logistic Regression.
- **`n_jobs`**: explicitly set to 4 for XGBoost only. Random Forest and Logistic Regression also expose parallelism options, but those are not currently set here and do not change the models' statistical objective.

Other important settings are model-specific: Logistic Regression uses `max_iter` and `solver`; Random Forest has tree controls such as `max_depth` and `min_samples_leaf` (currently left at defaults); XGBoost uses boosting and sampling controls such as `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree`. Do not describe these as equivalent knobs across all three algorithms. If hyperparameter tuning is added, keep the held-out test set for final evaluation and perform tuning only within the training data; document each search space and selected configuration.

Each notebook's final controlled benchmark cell imports `benchmark_lime()` and `evaluate_predictive_performance()`, then writes `{model}_controlled_lime_raw.csv` and `{model}_controlled_lime_summary.csv` to the notebook's current working directory. Keep the model names consistent (`XGBoost`, `LogisticRegression`, `RandomForest`) so output labels align.

## Run preparation and execution checklist

1. Inspect `experiments/controlled_lime_protocol.py` and each notebook's model-fit cell. Confirm all notebooks import the shared helper and use the model configurations above.
2. Confirm the notebook actually runs the controlled loader and does not later replace `X`, `y`, `X_train`, `X_test`, `y_train`, or `y_test` with a different dataset sample/split.
3. Check the notebook kernel has compatible `numpy`, `pandas`, `scipy`, `scikit-learn`, `lime`, and (for XGBoost) `xgboost`. The notebooks contain package-install cells, but record exact versions in the run record; do not assume old notebook outputs reflect the current environment.
4. Run the model fit and controlled benchmark for each model. Preserve each raw CSV and summary CSV, plus a record of machine/runtime environment, execution date, package versions, and any errors or reruns.
5. Check that each summary has 45 rows (3 feature counts × 5 sample budgets × 3 models) after combining the three model files, and that raw explanation rows have 1,575 rows per model. The three raw files combined should contain 4,725 rows.
6. Check all model × feature count × sample budget groups have 21 cases and 105 explanation runs. Check the stability summaries have finite-value counts and investigate any NaNs before reporting means.
7. Make plots/tables from the saved CSVs, keeping runtime, fidelity, stability, and predictive performance clearly separated. Include uncertainty (e.g. standard deviation) and describe averaging over cases and repeated runs.

## Important notebook caveats to resolve

- **Saved notebook outputs are not reliable evidence of the controlled run.** At least the Logistic Regression notebook has saved output showing 464,809 training rows and 116,203 test rows, while the shared helper's intended dimensions are 8,000 and 2,000. The XGBoost notebook also has an older cell that separately calls `train_test_split`; although it is applied to the helper's 10,000-row sample with the same seed, keep the helper split as the sole definition to avoid drift. Clear or rerun stale notebook outputs before using them in figures or the paper.
- Older notebook sections benchmark only one row, use different LIME feature counts, or compare sample budgets without the 18/36/54 sweep. These are exploratory demonstrations, not results for the controlled comparison.
- Earlier examples sometimes timed explainer construction or used all training rows as LIME background. Those timings/background choices differ from the controlled protocol and must not be mixed with its numbers.
- In `randomForest.ipynb`, an exploratory cell measures explainer-construction time and earlier cells use a single test row. The final controlled benchmark is the relevant cell.
- Covertype is imbalanced. Use stratification and report macro/balanced predictive metrics as well as accuracy. The 21 LIME cases are balanced by true class, unlike the full test set; be explicit that LIME diagnostic means are averaged over this balanced set of selected cases.
- LIME perturbations can create combinations of binary wilderness/soil indicators that may not occur in the original data. This is a limitation when interpreting fidelity and stability.

## Project context from logs

- The agreed paper topic is “LIME Cost/Quality Scaling on High-Dimensional Tabular Data using multiple different models” (log entry `04-10-2026.md`). The stated goal is to study how explanation runtime relates to stability and local fidelity as explanation feature-space size increases across models.
- The 7 October notes identify runtime, stability, and fidelity as the target measures; the shared protocol now specifies how those are operationalized.
- The 17 October notes say experiments need to be rerun for updated figures. The work breakdown assigns Sebastian to experiments and working with Amelia; Hugh to analysis/results, Amelia to methodology, and David to background/introduction. Treat assignments as context, not a requirement to contact anyone.
- Older notes contain paper research and unrelated ideas. They are intentionally not reproduced here because they are not needed to run the experiments.

## Files to start with

- `experiments/controlled_lime_protocol.py` — shared data, sampling, LIME, stability, and metric implementation.
- `experiments/coverType_XGBoost_activity.ipynb` — XGBoost setup and controlled benchmark call.
- `experiments/CovTypeLogRegLimeExp.ipynb` — Logistic Regression setup and controlled benchmark call.
- `experiments/randomForest.ipynb` — Random Forest setup and controlled benchmark call.
- `log_entries/07-10-2026.md`, `log_entries/04-10-2026.md`, and `log_entries/17-10-2026.md` — measurement intent, paper objective, and current rerun status.

No experiment was run while preparing this handoff.
