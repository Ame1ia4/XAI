# LIME model comparison — 8 October 2026

> **Historical results:** see the [9 October controlled rerun](LIME_comparison_report_09-10-2026.md) for the new results. The experiment settings differ.

**Random Forest produced the highest local fidelity and most stable explanations; Logistic Regression was fastest; XGBoost fell between them at a 5,000-sample budget.** These findings apply to the fitted models and observations tested here.

## Execution

All four notebooks in `experiments/` were attempted. Three completed, including their existing LIME experiments. Executed copies and fresh measurements are saved in [results/2026-10-08](results/2026-10-08/); original notebooks were preserved.

| Model | Training / test rows | Test accuracy | Outcome |
|---|---:|---:|---|
| Random Forest | 464,809 / 116,203 | 95.41% | Completed |
| Logistic Regression | 464,809 / 116,203 | 72.35% | Completed |
| XGBoost | 8,000 / 2,000 | 79.75% | Completed |
| TabFM | Planned: 3,000 / 1,000 | Unavailable | Failed before fitting |

TabFM failed at `import torch`. Its installer was deliberately interrupted after confirming the [package requires Python 3.11+](https://github.com/google-research/tabfm/blob/main/pyproject.toml); this project uses Python 3.10. The notebook also requests CUDA, unavailable on this Mac. A [CPU backend](https://github.com/google-research/tabfm#installation) would need a separate configuration. No TabFM metrics are claimed.

## Comparable LIME results

The original notebooks use different observations, settings, and stability definitions. A supplementary comparison standardized LIME using:

- Three shared held-out rows: **438215, 578257, 366706** (classes 1, 3, 2), checked against each model's training data.
- One shared 8,000-row background, categorical binary indicators, discretized continuous features, and 10 explanation features.
- **500, 1,000, 2,500, 5,000, 10,000** perturbations, five repetitions per row/budget, and reproducible advancing random states starting at seed 42.

Runtime covers `explain_instance`, excluding training and explainer setup. Fidelity is weighted local surrogate **R² (`exp.score`)**, not accuracy. Stability averages repetition pairs within each row: **top-five agreement** is shared features divided by five; **Spearman** compares signed weights over the union of selected features.

At **5,000 perturbations**, averaged over 15 explanations per model:

| Model | Runtime, seconds (mean ± SD) | Fidelity R² (mean ± SD) | Top-five agreement | Spearman |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0373 ± 0.0152 | 0.1951 ± 0.0616 | 0.4733 | 0.5794 |
| XGBoost | 0.0604 ± 0.0068 | 0.2957 ± 0.0131 | 0.6067 | 0.5042 |
| Random Forest | 0.1525 ± 0.0018 | 0.5557 ± 0.1101 | 0.8333 | 0.8101 |

SD includes observation and repetition variation. XGBoost beat Logistic Regression on feature overlap, but had lower signed-weight rank correlation.

![Runtime, fidelity and top-five stability across budgets](results/2026-10-08/comparison.png)

Larger budgets generally cost more without consistent fidelity gains. Doubling Random Forest's budget to 10,000 increased runtime to **0.2800 seconds**, left fidelity near **0.555**, and slightly reduced top-five agreement to **0.8200**; Spearman improved to **0.9112**. A 5,000-sample budget gave a useful cost/feature-overlap balance here.

## Original notebook results

Fresh original single-row results at 5,000 samples, with consistently recalculated feature overlap:

| Model | Explanation features | Runtime (seconds) | Fidelity R² | Top-five agreement |
|---|---:|---:|---:|---:|
| Logistic Regression | 10 | 0.0347 | 0.2782 | 0.6600 |
| XGBoost | 10 | 0.0556 | 0.3069 | 0.5400 |
| Random Forest | 54 | 0.2156 | 0.4113 | 0.8200 |

The differing scores show why observation and LIME settings matter. Even Random Forest's strongest mean fidelity is moderate.

## Interpretation and limits

**Random Forest gives the strongest fidelity and feature consistency at greater cost; Logistic Regression gives the cheapest explanations; XGBoost gives intermediate runtime and fidelity.** More perturbations alone did not resolve weak fidelity or feature instability.

This small comparison does not establish a model-family ranking or statistical significance. XGBoost's smaller training sample confounds model comparisons. LIME can generate unrealistic combinations of binary indicators, and fidelity describes agreement on its sampled neighbourhood.

Runs were sequential on Apple Silicon/16 GiB RAM, using scikit-learn 1.7.2, LIME 0.2.0.1, and XGBoost 3.2.0. Existing dependencies were retained and NumPy seeded at 42. The [manifest](results/2026-10-08/execution_manifest.json) records versions, adaptations, errors, timings, and unchanged source hashes. Notebook wall times include setup, plots, and supplementary measurements.

All budgets: [common_summary.csv](results/2026-10-08/common_summary.csv), [native_summary.csv](results/2026-10-08/native_summary.csv). Reproduce with `.venv/bin/python experiments/run_comparison.py`, then `.venv/bin/python experiments/summarize_comparison.py`. The runner now checks TabFM requirements before setup.
