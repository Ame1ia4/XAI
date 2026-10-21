"""Combine saved per-model results and draw the paper figures.

Usage (from the repository root): python -m lime_study.plot_results [--quick] [--variant NAME]
Reads only the CSV/JSON files in results/, so figures can be redrawn without rerunning.
Bands are bootstrap CIs over explained cases. Settings with N <= d are drawn as hollow,
unconnected markers and kept out of the trend lines.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from lime_study import analysis, config  # noqa: E402

# Validated categorical palette (fixed order) plus marker shapes as secondary encoding.
MODEL_STYLE = {
    "XGBoost": {"color": "#2a78d6", "marker": "o"},
    "LogisticRegression": {"color": "#eb6834", "marker": "s"},
    "RandomForest": {"color": "#1baf7a", "marker": "^"},
}
MODEL_LABELS = {
    "XGBoost": "XGBoost",
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
}
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FULL_WIDTH_IN = 7.16  # IEEE two-column figure* width

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": "#0b0b0b",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_case_level(sweep: config.Sweep, variant: config.Variant) -> pd.DataFrame:
    results = config.results_dir(sweep, variant)
    frames = []
    for model in config.MODEL_NAMES:
        raw_path, stability_path = results / f"{model}_raw.csv", results / f"{model}_stability.csv"
        if raw_path.exists() and stability_path.exists():
            frames.append(analysis.case_level(pd.read_csv(raw_path), pd.read_csv(stability_path)))
        else:
            print(f"WARNING: missing results for {model} in {results}; left out of the figures")
    if not frames:
        raise FileNotFoundError(f"No *_raw.csv files in {results}")
    cases = pd.concat(frames, ignore_index=True)
    per_model = cases.groupby("model")["case_index"].apply(lambda s: tuple(sorted(set(s))))
    if per_model.nunique() != 1:
        raise AssertionError("Models were explained on different cases; paired comparison invalid")
    run_ids = {
        json.loads((results / f"{model}_run_meta.json").read_text()).get("run_id")
        for model in set(cases["model"])
    }
    meta = json.loads((results / f"{cases['model'].iloc[0]}_run_meta.json").read_text())
    mismatches = meta.get("version_check", {}).get("mismatches", [])
    if mismatches:
        print(f"WARNING: these results were produced outside the pinned environment: {mismatches}")
    if len(run_ids) != 1 or None in run_ids:
        raise AssertionError(
            f"Results in {results} come from different runs {run_ids}; models must be compared "
            "within one interleaved run. Rerun python -m lime_study.run_all."
        )
    return cases


def performance_table(sweep: config.Sweep, variant: config.Variant) -> pd.DataFrame:
    rows = []
    for model in config.MODEL_NAMES:
        path = config.results_dir(sweep, variant) / f"{model}_performance.json"
        if path.exists():
            rows.append({"model": model, **json.loads(path.read_text())})
    return pd.DataFrame(rows)


def drift_table(sweep: config.Sweep, variant: config.Variant) -> pd.DataFrame:
    """Speed probes through the timed phase; a slower machine shows up as a larger probe.

    All models share one interleaved run, so drift affects them alike; this trace
    documents how much drift there was.
    """
    for model in config.MODEL_NAMES:
        path = config.results_dir(sweep, variant) / f"{model}_run_meta.json"
        if path.exists():
            table = pd.DataFrame(json.loads(path.read_text())["speed_probes"])
            table["relative_to_first"] = table["seconds"] / table["seconds"].iloc[0]
            table["relative_to_median"] = table["seconds"] / table["seconds"].median()
            return table
    return pd.DataFrame()


def _by_model(data: pd.DataFrame):
    """Yield groups in the fixed config order so legend order never depends on the data."""
    for model in config.MODEL_NAMES:
        group = data[data["model"] == model]
        if not group.empty:
            yield model, group


def _yerr(data: pd.DataFrame, metric: str) -> list:
    mean = data[f"{metric}_mean"]
    # Clip float noise when the CI collapses onto the mean (e.g. one case per class).
    # Plain arrays: matplotlib calls float() on one-element Series (the single N <= d
    # point), which pandas deprecates.
    return [
        (mean - data[f"{metric}_ci_low"]).clip(lower=0).to_numpy(),
        (data[f"{metric}_ci_high"] - mean).clip(lower=0).to_numpy(),
    ]


def _plot_metric(ax, data: pd.DataFrame, metric: str) -> None:
    for model, group in _by_model(data):
        style = MODEL_STYLE[model]
        valid = group[~group["low_sample_ratio"]]
        flagged = group[group["low_sample_ratio"]]
        ax.plot(valid["num_samples"], valid[f"{metric}_mean"], label=MODEL_LABELS[model], **style)
        ax.fill_between(
            valid["num_samples"],
            valid[f"{metric}_ci_low"],
            valid[f"{metric}_ci_high"],
            color=style["color"],
            alpha=0.15,
            linewidth=0,
        )
        if not flagged.empty:
            ax.errorbar(
                flagged["num_samples"],
                flagged[f"{metric}_mean"],
                yerr=_yerr(flagged, metric),
                linestyle="none",
                marker=style["marker"],
                markerfacecolor="white",
                color=style["color"],
                elinewidth=0.8,
            )


def _format_budget_axis(ax, budgets: tuple[int, ...]) -> None:
    """Linear budget axis from 0, so spacing is proportional to the number of samples.

    Markers sit at the true budgets; ticks use round values because 500 and 1k
    would overlap if each budget were labelled.
    """
    ax.set_xlim(0, max(budgets) * 1.05)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}")
    )


def _save(fig, out_dir, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}")
    plt.close(fig)


COST_FIDELITY_ROWS = (
    ("runtime_seconds", "Runtime per\nexplanation (s)"),
    ("fidelity_heldout", "Held-out surrogate fit\n(weighted R², non-flat)"),
    ("heldout_rmse", "Held-out surrogate error\n(weighted RMSE, absolute)"),
)
STABILITY_ROWS = (
    ("reference_jaccard", f"Top-{config.TOP_K} Jaccard vs\nlarge-N reference"),
    ("reference_jaccard_relative", "Reference agreement /\nreference ceiling"),
    ("repeat_jaccard", f"Top-{config.TOP_K} Jaccard\nbetween repeats"),
)

# One figure answering the three headline questions at a glance: cost, fidelity, stability.
OVERVIEW_ROWS = (
    ("runtime_seconds", "Runtime per\nexplanation (s)"),
    ("fidelity_heldout", "Fidelity: held-out\nfit (weighted R²)"),
    ("repeat_jaccard", f"Stability: top-{config.TOP_K}\nJaccard between repeats"),
)


def _plot_ceiling(ax, data: pd.DataFrame) -> None:
    """How well the two large-N references agree (constant across budgets).

    Dashed: passes the reference rule; dotted: fails it, so agreement with the
    reference is not read as convergence there. Grey line: the rule's threshold.
    """
    ax.axhline(config.REFERENCE_MIN_AGREEMENT, color=TEXT_SECONDARY, linewidth=0.6, alpha=0.6)
    for model, group in _by_model(data):
        ax.axhline(
            group["reference_ceiling_mean"].iloc[0],
            color=MODEL_STYLE[model]["color"],
            linestyle="--" if group["reference_valid"].iloc[0] else ":",
            linewidth=0.9,
            alpha=0.8,
        )


def plot_grid(summary: pd.DataFrame, sweep: config.Sweep, out_dir, rows, name: str, note: str) -> None:
    fig, axes = plt.subplots(
        len(rows),
        len(config.FEATURE_COUNTS),
        figsize=(FULL_WIDTH_IN, 1.9 * len(rows) + 0.6),
        sharex=True,
        sharey="row",
    )
    for col, feature_count in enumerate(config.FEATURE_COUNTS):
        subset = summary[summary["feature_count"] == feature_count]
        for row, (metric, label) in enumerate(rows):
            ax = axes[row, col]
            _plot_metric(ax, subset, metric)
            if metric == "reference_jaccard":
                _plot_ceiling(ax, subset)
            _format_budget_axis(ax, sweep.sample_budgets)
            if row == 0:
                ax.set_title(f"{feature_count} perturbed features")
            if col == 0:
                ax.set_ylabel(label)
    if rows[0][0] == "runtime_seconds":
        axes[0, 0].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    # One shared x label: with four columns a label under each panel runs into its neighbour.
    # The footnote hangs below it (va="top"), so wrapped lines grow downwards, not into it.
    fig.text(0.5, 0.025, "LIME samples (num_samples)", ha="center", va="top")
    fig.text(
        0.5,
        -0.005,
        va="top",
        s=f"Bands: {config.CI_LEVEL:.0%} class-stratified bootstrap CI over explained cases. "
        f"Hollow markers: N ≤ d (low sample-to-feature ratio), kept out of the lines. {note}",
        ha="center",
        color=TEXT_SECONDARY,
        wrap=True,
    )
    _save(fig, out_dir, name)


def plot_cost_vs_quality(summary: pd.DataFrame, out_dir) -> None:
    fig, axes = plt.subplots(
        1, len(config.FEATURE_COUNTS), figsize=(FULL_WIDTH_IN, 2.6), sharex=True, sharey=True
    )
    for ax, feature_count in zip(axes, config.FEATURE_COUNTS):
        subset = summary[summary["feature_count"] == feature_count]
        for model, group in _by_model(subset):
            style = MODEL_STYLE[model]
            valid = group[~group["low_sample_ratio"]]
            flagged = group[group["low_sample_ratio"]]
            ax.errorbar(
                valid["runtime_seconds_mean"],
                valid["reference_jaccard_mean"],
                yerr=_yerr(valid, "reference_jaccard"),
                label=MODEL_LABELS[model],
                elinewidth=0.8,
                capsize=1.5,
                **style,
            )
            ax.plot(
                flagged["runtime_seconds_mean"],
                flagged["reference_jaccard_mean"],
                linestyle="none",
                marker=style["marker"],
                markerfacecolor="white",
                color=style["color"],
            )
        ax.set_title(f"{feature_count} perturbed features")
    # Shared axis: set the range once from all panels so no panel's data is clipped.
    axes[0].set_xlim(0, summary["runtime_seconds_mean"].max() * 1.05)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(f"Top-{config.TOP_K} Jaccard vs\nlarge-N reference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.0, "Mean runtime per explanation (s)", ha="center")
    fig.text(
        0.5,
        -0.08,
        "Points along each line are sample budgets, increasing left to right. "
        "Error bars: bootstrap CI over cases; hollow markers: N ≤ d.",
        ha="center",
        color=TEXT_SECONDARY,
    )
    _save(fig, out_dir, "cost_vs_quality")


def check_same_models_across_variants(sweep: config.Sweep) -> None:
    """Every variant must explain the same fitted models on the same cases.

    Variants are separate runs that refit each model; fixed seeds and single threading
    should make the fits identical, and the saved fingerprints prove it (or fail here).
    """
    seen: dict[str, tuple[str, str, tuple]] = {}
    for variant in config.VARIANTS.values():
        for model in config.MODEL_NAMES:
            path = config.results_dir(sweep, variant) / f"{model}_run_meta.json"
            if not path.exists():
                continue
            meta = json.loads(path.read_text())
            if "model_fingerprint" not in meta:
                raise AssertionError(f"{path} predates model fingerprints; rerun that variant")
            record = (meta["model_fingerprint"], tuple(meta["explained_cases"]))
            if model in seen and seen[model][1:] != record:
                raise AssertionError(
                    f"{model} differs between variants {seen[model][0]} and {variant.name} "
                    "(different fitted model or explained cases); comparisons across variants are invalid"
                )
            seen.setdefault(model, (variant.name, *record))


def main(quick: bool = False, variant_name: str = config.MAIN_VARIANT.name) -> None:
    sweep = config.QUICK_SWEEP if quick else config.FULL_SWEEP
    variant = config.VARIANTS[variant_name]
    out_dir = config.figures_dir(sweep, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_case_level(sweep, variant)
    check_same_models_across_variants(sweep)
    summary = analysis.summarise(cases)
    expected = len(config.FEATURE_COUNTS) * len(sweep.sample_budgets) * cases["model"].nunique()
    if len(summary) != expected:
        raise AssertionError(f"Expected {expected} combined summary rows, got {len(summary)}")
    summary.to_csv(out_dir / "combined_summary.csv", index=False)
    analysis.paired_differences(cases).to_csv(out_dir / "paired_differences.csv", index=False)
    analysis.reference_check(cases).to_csv(out_dir / "reference_check.csv", index=False)
    analysis.runtime_scaling(cases).to_csv(out_dir / "runtime_scaling.csv", index=False)
    performance_table(sweep, variant).to_csv(out_dir / "performance_table.csv", index=False)
    drift_table(sweep, variant).to_csv(out_dir / "machine_drift.csv", index=False)
    plot_grid(
        summary, sweep, out_dir, COST_FIDELITY_ROWS, "lime_scaling_grid",
        f"R² omits flat cases (output SD < {config.FLAT_OUTPUT_SD}).",
    )
    plot_grid(
        summary, sweep, out_dir, STABILITY_ROWS, "lime_stability_grid",
        "Dashed: agreement between the two large-N references (ceiling); dotted: below the "
        f"pre-set {config.REFERENCE_MIN_AGREEMENT:.3f} rule (grey line), so not a valid reference.",
    )
    plot_grid(
        summary, sweep, out_dir, OVERVIEW_ROWS, "lime_overview",
        f"R² omits flat cases (output SD < {config.FLAT_OUTPUT_SD}). Details in "
        "lime_scaling_grid (adds RMSE) and lime_stability_grid (adds the large-N references).",
    )
    plot_cost_vs_quality(summary, out_dir)
    print(f"Figures and tables written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw figures from saved LIME results")
    parser.add_argument("--quick", action="store_true", help="plot the smoke-test results")
    parser.add_argument("--variant", choices=list(config.VARIANTS), default=config.MAIN_VARIANT.name)
    args = parser.parse_args()
    main(quick=args.quick, variant_name=args.variant)
