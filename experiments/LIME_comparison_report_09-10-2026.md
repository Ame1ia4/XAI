# LIME model comparison — 9 October 2026 controlled rerun

**Provisional: TabFM is still executing. This report will update automatically when that attempt ends.**

**These are new results from the shared controlled protocol.** The [8 October report](LIME_comparison_report_08-10-2026.md) and its results are historical; they used different training sizes and explanation settings.

## Execution and method

Each completed model used the same stratified 10,000-row Covertype sample: **8,000 training rows, 2,000 test rows, and 54 input predictors**. LIME used a shared 2,000-row training background and **21 fixed held-out cases (three per true class)**, selected without filtering for correct predictions.

The benchmark varied perturbation dimensions **18/36/54** and sample budgets **500/1,000/2,500/5,000/10,000**, with five repeats seeded 42–46 and ten explanation terms. That gives **1,575 explanations per completed model**. All models remain fitted on all 54 predictors; dimension changes affect LIME perturbations only.

Runtime covers the explanation call, including model predictions and surrogate fitting, but excludes explainer setup and prediction warm-up. Fidelity is weighted local surrogate **R² (`exp.score`)**, not classification accuracy. Stability compares repeat pairs within each case: signed-weight **Spearman** over selected-feature unions and **top-five Jaccard** (intersection/union).

## Main comparison

At **54 perturbed features and 5,000 samples**, means cover 105 explanations per model. SD includes variation between cases and repeats.

| Model | Runtime seconds (mean ± SD) | Fidelity R² (mean ± SD) | Spearman | Top-five Jaccard |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0267 ± 0.0034 | 0.2268 ± 0.0833 | 0.5263 | 0.3340 |
| Random Forest | 0.0503 ± 0.0036 | 0.4369 ± 0.1966 | 0.8450 | 0.7831 |
| XGBoost | 0.0563 ± 0.0082 | 0.1980 ± 0.0905 | 0.4875 | 0.4038 |
| TabFM | Unavailable | Unavailable | Unavailable | Unavailable |

**Logistic Regression was fastest; Random Forest had the highest mean fidelity; Random Forest had the highest top-five feature stability** at this comparison point.

Random Forest's mean R² is still moderate: consistent feature selection does not by itself establish a faithful explanation.

![Controlled runtime, fidelity and stability across budgets](results/2026-10-09/comparison_54_features.png)

## Scaling and predictive performance

- **Logistic Regression**, 500 → 10,000 samples: runtime 0.0081 → 0.0471s; fidelity 0.253 → 0.231; top-five Jaccard 0.236 → 0.400.
- **Random Forest**, 500 → 10,000 samples: runtime 0.0215 → 0.0926s; fidelity 0.432 → 0.439; top-five Jaccard 0.606 → 0.878.
- **XGBoost**, 500 → 10,000 samples: runtime 0.0134 → 0.1077s; fidelity 0.232 → 0.193; top-five Jaccard 0.260 → 0.433.

Higher-dimensional perturbations expose more of the fitted model's behaviour; scores at 18 and 54 dimensions therefore measure different explanation neighbourhoods. Compare models at matching dimensions and budgets.

| Model | Test accuracy | Macro F1 | Notebook wall time |
|---|---:|---:|---:|
| Logistic Regression | 72.05% | 0.5046 | 83.8s |
| XGBoost | 79.75% | 0.6730 | 113.2s |
| Random Forest | 80.70% | 0.6932 | 113.5s |

Notebook wall time includes fitting, cross-validation or permutation importance where present, illustrations, and the benchmark; it is not comparable training time.

## Limits and provenance

These results describe 21 cases and the chosen model configurations, not an intrinsic model-family ranking. Explanations target each model's predicted class, which can differ between models. Independent binary perturbations can create unrealistic soil/wilderness combinations, and local R² is measured on LIME's sampled neighbourhood rather than independent validation data.

Runs were sequential on an Apple Silicon Mac with 16 GiB RAM using Python 3.13.7, scikit-learn 1.7.2, LIME 0.2.0.1, and XGBoost 3.2.0. CPU thread limits were four. Progress checkpointing occurred outside timed calls.

The [new result directory](results/2026-10-09/) contains executed notebooks, raw measurements, all 15 summary groups per model, pairwise stability results, and the execution manifest. Pre-rerun root CSVs were copied into `before-rerun/`; root CSVs now contain the latest completed results. The 8 October data remain in `results/2026-10-08/`. Because controls and the Python environment changed, old/new score differences should not be interpreted as improvements or regressions.

## Incomplete execution

**TabFM:** running; 0 completed benchmark explanations recorded so far. Execution details are retained in the manifest and executed notebook. No completed-model scores are attributed to this attempt.
