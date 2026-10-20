"""Combine saved per-model results and draw the paper figures.

Usage (from the repository root): python -m lime_study.plot_results [--quick]
Reads only the CSV/JSON files in results/, so figures can be redrawn without rerunning.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from lime_study import config  # noqa: E402

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


def load_combined_summary(sweep: config.Sweep) -> pd.DataFrame:
    results = config.results_dir(sweep)
    frames = []
    for model in config.MODEL_NAMES:
        path = results / f"{model}_summary.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: missing {path}; {model} left out of the figures")
    if not frames:
        raise FileNotFoundError(f"No *_summary.csv files in {results}")
    combined = pd.concat(frames, ignore_index=True)
    expected = len(config.FEATURE_COUNTS) * len(sweep.sample_budgets) * len(frames)
    if len(combined) != expected:
        raise AssertionError(f"Expected {expected} combined summary rows, got {len(combined)}")
    return combined.sort_values(["model", "feature_count", "num_samples"])


def performance_table(sweep: config.Sweep) -> pd.DataFrame:
    rows = []
    for model in config.MODEL_NAMES:
        path = config.results_dir(sweep) / f"{model}_performance.json"
        if path.exists():
            rows.append({"model": model, **json.loads(path.read_text())})
    return pd.DataFrame(rows)


def _by_model(data: pd.DataFrame):
    """Yield groups in the fixed config order so legend order never depends on the data."""
    for model in config.MODEL_NAMES:
        group = data[data["model"] == model]
        if not group.empty:
            yield model, group


def _plot_metric(ax, data: pd.DataFrame, mean: str, std: str | None) -> None:
    for model, group in _by_model(data):
        style = MODEL_STYLE[model]
        ax.plot(group["num_samples"], group[mean], label=MODEL_LABELS[model], **style)
        if std is not None:
            ax.fill_between(
                group["num_samples"],
                group[mean] - group[std],
                group[mean] + group[std],
                color=style["color"],
                alpha=0.15,
                linewidth=0,
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


def plot_scaling_grid(summary: pd.DataFrame, sweep: config.Sweep, out_dir) -> None:
    rows = [
        ("runtime_mean_seconds", "runtime_std_seconds", "Runtime per\nexplanation (s)"),
        ("fidelity_mean", "fidelity_std", "Local surrogate\nfit (R²)"),
        ("jaccard_mean", "jaccard_std", f"Top-{config.TOP_K} Jaccard\nstability"),
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(config.FEATURE_COUNTS),
        figsize=(FULL_WIDTH_IN, 6.0),
        sharex=True,
        sharey="row",
    )
    for col, feature_count in enumerate(config.FEATURE_COUNTS):
        subset = summary[summary["feature_count"] == feature_count]
        for row, (mean, std, label) in enumerate(rows):
            ax = axes[row, col]
            _plot_metric(ax, subset, mean, std)
            _format_budget_axis(ax, sweep.sample_budgets)
            if row == 0:
                ax.set_title(f"{feature_count} perturbed features")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(rows) - 1:
                ax.set_xlabel("LIME samples (num_samples)")
    axes[0, 0].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, "lime_scaling_grid")


def plot_cost_vs_quality(summary: pd.DataFrame, out_dir) -> None:
    fig, axes = plt.subplots(
        1, len(config.FEATURE_COUNTS), figsize=(FULL_WIDTH_IN, 2.4), sharex=True, sharey=True
    )
    for ax, feature_count in zip(axes, config.FEATURE_COUNTS):
        subset = summary[summary["feature_count"] == feature_count]
        for model, group in _by_model(subset):
            ax.plot(
                group["runtime_mean_seconds"],
                group["jaccard_mean"],
                label=MODEL_LABELS[model],
                **MODEL_STYLE[model],
            )
        ax.set_title(f"{feature_count} perturbed features")
        ax.set_xlabel("Mean runtime per explanation (s)")
    # Shared axis: set the range once from all panels so no panel's data is clipped.
    axes[0].set_xlim(0, summary["runtime_mean_seconds"].max() * 1.05)
    axes[0].set_ylabel(f"Top-{config.TOP_K} Jaccard stability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.text(
        0.5,
        -0.04,
        "Points along each line are sample budgets, increasing left to right.",
        ha="center",
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    _save(fig, out_dir, "cost_vs_quality")


def main(quick: bool = False) -> None:
    sweep = config.QUICK_SWEEP if quick else config.FULL_SWEEP
    out_dir = config.figures_dir(sweep)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_combined_summary(sweep)
    summary.to_csv(out_dir / "combined_summary.csv", index=False)
    performance_table(sweep).to_csv(out_dir / "performance_table.csv", index=False)
    plot_scaling_grid(summary, sweep, out_dir)
    plot_cost_vs_quality(summary, out_dir)
    print(f"Figures and tables written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw figures from saved LIME results")
    parser.add_argument("--quick", action="store_true", help="plot the smoke-test results")
    main(quick=parser.parse_args().quick)
