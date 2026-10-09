"""Validate the 9 October controlled rerun and write its brief comparison report."""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/2026-10-09"
manifest = json.loads((OUT / "execution_manifest.json").read_text())
completed = [r for r in manifest["runs"] if r["status"] == "completed"]
pretty = {"RandomForest":"Random Forest","LogisticRegression":"Logistic Regression",
          "XGBoost":"XGBoost","TabFM":"TabFM"}
raw_tables, summary_tables, predictive_rows, pair_records = [], [], [], []
reference = None
for record in completed:
    name = record["model"]
    raw = pd.read_csv(OUT / f"{name.lower()}_controlled_lime_raw.csv")
    summary = pd.read_csv(OUT / f"{name.lower()}_controlled_lime_summary.csv")
    context = json.loads((OUT / f"{name.lower()}_context.json").read_text())
    assert len(raw) == 1575 and len(summary) == 15, name
    assert raw.case_index.nunique() == 21
    assert raw.groupby("true_class").case_index.nunique().eq(3).all()
    assert raw.groupby(["case_index","feature_count","num_samples"]).size().eq(5).all()
    assert raw.runtime_seconds.gt(0).all() and np.isfinite(raw.local_fidelity_score).all()
    assert summary.cases.eq(21).all() and summary.explanation_runs.eq(105).all()
    assert context["train_rows"] == 8000 and context["test_rows"] == 2000
    assert context["input_features"] == 54 and context["background_rows"] == 2000
    if reference is None: reference = context["data_fingerprints"]
    assert reference == context["data_fingerprints"], "Model data/background/cases differ"
    events = [json.loads(line) for line in (OUT / f"{name.lower()}_checkpoint.jsonl").read_text().splitlines()]
    assert len(events) == len(raw)
    grouped = {}
    for event in events:
        observation = event["observation"]
        key = tuple(observation[k] for k in ["case_index","feature_count","num_samples"])
        grouped.setdefault(key,[]).append(event)
    for (case,features,budget),events_for_case in grouped.items():
        assert sorted(e["observation"]["repeat"] for e in events_for_case) == [1,2,3,4,5]
        for first,second in combinations(events_for_case,2):
            a,b = np.array(first["weights"]),np.array(second["weights"])
            selected = np.flatnonzero((a != 0) | (b != 0))
            rho = spearmanr(a[selected],b[selected]).statistic if len(selected)>=2 else np.nan
            ai,bi = np.flatnonzero(a),np.flatnonzero(b)
            at = set(ai[np.argsort(np.abs(a[ai]))[-5:]])
            bt = set(bi[np.argsort(np.abs(b[bi]))[-5:]])
            union = at | bt
            pair_records.append({"model":name,"case_index":case,"feature_count":features,
                "num_samples":budget,"first_repeat":first["observation"]["repeat"],
                "second_repeat":second["observation"]["repeat"],"spearman":rho,
                "top5_jaccard":len(at & bt)/len(union) if union else np.nan})
    predictive_rows.append({"model":name,**context["predictive_metrics"],
        "notebook_wall_seconds":record["notebook_wall_seconds"]})
    raw_tables.append(raw)
    summary_tables.append(summary)

raw = pd.concat(raw_tables,ignore_index=True)
summary = pd.concat(summary_tables,ignore_index=True)
pairs = pd.DataFrame(pair_records)
assert len(pairs) == 3150*len(completed)
recomputed = pairs.groupby(["model","feature_count","num_samples"])[["spearman","top5_jaccard"]].mean()
for row in summary.itertuples():
    metrics = recomputed.loc[(row.model,row.feature_count,row.num_samples)]
    assert np.isclose(row.spearman_stability_mean,metrics.spearman,equal_nan=True)
    assert np.isclose(row.top5_jaccard_mean,metrics.top5_jaccard,equal_nan=True)
    observations = raw[(raw.model==row.model)&(raw.feature_count==row.feature_count)&(raw.num_samples==row.num_samples)]
    assert np.isclose(row.runtime_mean_seconds,observations.runtime_seconds.mean())
    assert np.isclose(row.fidelity_mean,observations.local_fidelity_score.mean())
raw.to_csv(OUT / "controlled_lime_raw.csv",index=False)
summary.to_csv(OUT / "controlled_lime_summary.csv",index=False)
pairs.to_csv(OUT / "controlled_stability_pairs.csv",index=False)
predictive = pd.DataFrame(predictive_rows)
predictive.to_csv(OUT / "predictive_metrics.csv",index=False)

full = summary[summary.feature_count==54]
fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
for model,table in full.groupby("model"):
    table=table.sort_values("num_samples")
    axes[0,0].errorbar(table.num_samples,table.runtime_mean_seconds,yerr=table.runtime_std_seconds,
                      marker="o",capsize=3,label=pretty[model])
    axes[0,1].errorbar(table.num_samples,table.fidelity_mean,yerr=table.fidelity_std,
                      marker="o",capsize=3,label=pretty[model])
    axes[1,0].plot(table.num_samples,table.spearman_stability_mean,marker="o",label=pretty[model])
    axes[1,1].plot(table.num_samples,table.top5_jaccard_mean,marker="o",label=pretty[model])
for ax,title,label in zip(axes.flat,["Explanation runtime","Local fidelity","Signed-weight rank stability","Top-five feature stability"],
    ["Seconds (mean ± SD)","Weighted surrogate R² (mean ± SD)","Mean pairwise Spearman","Mean pairwise Jaccard"]):
    ax.set(title=title,xlabel="Perturbation samples",ylabel=label,xscale="log")
    ax.set_xticks([500,1000,2500,5000,10000],labels=["500","1k","2.5k","5k","10k"])
    ax.grid(alpha=.25)
axes[0,0].set_ylim(bottom=0)
if full.runtime_mean_seconds.max() / full.runtime_mean_seconds.min() > 30:
    axes[0,0].set_yscale("log")
    axes[0,0].set_ylim(bottom=full.runtime_mean_seconds.min()/2)
axes[0,0].legend(fontsize=9)
axes[1,0].set_ylim(-.05,1.05)
axes[1,1].set_ylim(-.05,1.05)
fig.suptitle("9 October controlled rerun — 54 perturbed features, 21 shared cases")
fig.savefig(OUT / "comparison_54_features.png",dpi=180)

primary = full[full.num_samples==5000].sort_values("runtime_mean_seconds")
primary.to_csv(OUT / "comparison_54_features_5000_samples.csv",index=False)
lines = ["# LIME model comparison — 9 October 2026 controlled rerun", "",
    "**These are new results from the shared controlled protocol.** The [8 October report](LIME_comparison_report_08-10-2026.md) and its results are historical; they used different training sizes and explanation settings.","",
    "## Execution and method", "",
    "Each completed model used the same stratified 10,000-row Covertype sample: **8,000 training rows, 2,000 test rows, and 54 input predictors**. LIME used a shared 2,000-row training background and **21 fixed held-out cases (three per true class)**, selected without filtering for correct predictions.", "",
    "The benchmark varied perturbation dimensions **18/36/54** and sample budgets **500/1,000/2,500/5,000/10,000**, with five repeats seeded 42–46 and ten explanation terms. That gives **1,575 explanations per completed model**. All models remain fitted on all 54 predictors; dimension changes affect LIME perturbations only.", "",
    "Runtime covers the explanation call, including model predictions and surrogate fitting, but excludes explainer setup and prediction warm-up. Fidelity is weighted local surrogate **R² (`exp.score`)**, not classification accuracy. Stability compares repeat pairs within each case: signed-weight **Spearman** over selected-feature unions and **top-five Jaccard** (intersection/union).", "",
    "## Main comparison", "", "At **54 perturbed features and 5,000 samples**, means cover 105 explanations per model. SD includes variation between cases and repeats.", "",
    "| Model | Runtime seconds (mean ± SD) | Fidelity R² (mean ± SD) | Spearman | Top-five Jaccard |",
    "|---|---:|---:|---:|---:|"]
if any(r["status"] == "running" for r in manifest["runs"]):
    lines.insert(2,"**Provisional: TabFM is still executing. This report will update automatically when that attempt ends.**\n")
for row in primary.itertuples():
    lines.append(f"| {pretty[row.model]} | {row.runtime_mean_seconds:.4f} ± {row.runtime_std_seconds:.4f} | {row.fidelity_mean:.4f} ± {row.fidelity_std:.4f} | {row.spearman_stability_mean:.4f} | {row.top5_jaccard_mean:.4f} |")
for record in manifest["runs"]:
    if record["status"] != "completed": lines.append(f"| {pretty[record['model']]} | Unavailable | Unavailable | Unavailable | Unavailable |")
fastest = primary.loc[primary.runtime_mean_seconds.idxmin()]
best_fit = primary.loc[primary.fidelity_mean.idxmax()]
best_overlap = primary.loc[primary.top5_jaccard_mean.idxmax()]
lines += ["",f"**{pretty[fastest.model]} was fastest; {pretty[best_fit.model]} had the highest mean fidelity; {pretty[best_overlap.model]} had the highest top-five feature stability** at this comparison point.","",
    "Random Forest's mean R² is still moderate: consistent feature selection does not by itself establish a faithful explanation.","",
    "![Controlled runtime, fidelity and stability across budgets](results/2026-10-09/comparison_54_features.png)","",
    "## Scaling and predictive performance",""]
for model,table in full.groupby("model"):
    low=table[table.num_samples==500].iloc[0]
    high=table[table.num_samples==10000].iloc[0]
    lines.append(f"- **{pretty[model]}**, 500 → 10,000 samples: runtime {low.runtime_mean_seconds:.4f} → {high.runtime_mean_seconds:.4f}s; fidelity {low.fidelity_mean:.3f} → {high.fidelity_mean:.3f}; top-five Jaccard {low.top5_jaccard_mean:.3f} → {high.top5_jaccard_mean:.3f}.")
lines += ["", "Higher-dimensional perturbations expose more of the fitted model's behaviour; scores at 18 and 54 dimensions therefore measure different explanation neighbourhoods. Compare models at matching dimensions and budgets.","",
    "| Model | Test accuracy | Macro F1 | Notebook wall time |", "|---|---:|---:|---:|"]
for row in predictive.itertuples():
    lines.append(f"| {pretty[row.model]} | {100*row.accuracy:.2f}% | {row.macro_f1:.4f} | {row.notebook_wall_seconds:.1f}s |")
lines += ["", "Notebook wall time includes fitting, cross-validation or permutation importance where present, illustrations, and the benchmark; it is not comparable training time.","",
    "## Limits and provenance", "",
    "These results describe 21 cases and the chosen model configurations, not an intrinsic model-family ranking. Explanations target each model's predicted class, which can differ between models. Independent binary perturbations can create unrealistic soil/wilderness combinations, and local R² is measured on LIME's sampled neighbourhood rather than independent validation data.","",
    f"Runs were sequential on an Apple Silicon Mac with 16 GiB RAM using Python {manifest['python'].split()[0]}, scikit-learn {manifest['packages']['scikit-learn']}, LIME {manifest['packages']['lime']}, and XGBoost {manifest['packages']['xgboost']}. CPU thread limits were four. Progress checkpointing occurred outside timed calls.","",
    "The [new result directory](results/2026-10-09/) contains executed notebooks, raw measurements, all 15 summary groups per model, pairwise stability results, and the execution manifest. Pre-rerun root CSVs were copied into `before-rerun/`; root CSVs now contain the latest completed results. The 8 October data remain in `results/2026-10-08/`. Because controls and the Python environment changed, old/new score differences should not be interpreted as improvements or regressions."]
failures=[r for r in manifest["runs"] if r["status"] != "completed"]
if failures:
    lines += ["", "## Incomplete execution", ""]
    for record in failures:
        lines.append(f"**{pretty[record['model']]}:** {record['status']}; {record.get('checkpoint_explanations',0)} completed benchmark explanations recorded so far. Execution details are retained in the manifest and executed notebook. No completed-model scores are attributed to this attempt.")
report=ROOT / "experiments/LIME_comparison_report_09-10-2026.md"
report.write_text("\n".join(lines)+"\n")
print(primary.round(4).to_string(index=False))
print(predictive.round(4).to_string(index=False))
print(f"Validated {len(completed)} completed models, {len(raw)} explanations and {len(pairs)} stability pairs.")
print('Report:',report)
