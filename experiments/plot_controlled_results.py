"""Shared graphs for the controlled LIME cost/quality benchmark.

Every model notebook writes ``<model>_controlled_lime_summary.csv`` via
``controlled_lime_protocol.benchmark_lime``. This module turns those CSVs into the
same set of graphs for every model, so the four experiments are plotted identically.

From a notebook:
    from plot_controlled_results import plot_model_results, plot_model_comparison
    plot_model_results(controlled_summary, "TabFM")         # one model
    plot_model_comparison()                                  # every summary CSV found

From a terminal (inside the experiments folder):
    python plot_controlled_results.py            # plots every *_controlled_lime_summary.csv here
    python plot_controlled_results.py --dir path/to/csvs --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

SUMMARY_SUFFIX = "_controlled_lime_summary.csv"

# Fixed model -> colour/marker assignment, so a model looks the same in every graph
# regardless of which other models' CSVs are present.
MODEL_ORDER = ("LogisticRegression", "RandomForest", "XGBoost", "TabFM")
MODEL_COLOURS = {
    "LogisticRegression": "#2a78d6",
    "RandomForest": "#eb6834",
    "XGBoost": "#1baf7a",
    "TabFM": "#eda100",
}
EXTRA_COLOURS = ("#e87ba4", "#008300")
MODEL_MARKERS = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^", "TabFM": "D"}

# Feature count is ordered (18 < 36 < 54), so it uses one hue light -> dark.
FEATURE_COUNT_COLOURS = ("#6da7ec", "#2a78d6", "#104281")
FEATURE_COUNT_MARKERS = ("o", "s", "^")

INK = "#2b2b29"
MUTED_INK = "#6b6a64"
GRID = "#e4e3de"

METRICS = {
    "runtime": {
        "column": "runtime_mean_seconds",
        "spread": "runtime_std_seconds",
        "label": "Seconds per explanation",
        "title": "Runtime cost",
        "log_y": True,
    },
    "fidelity": {
        "column": "fidelity_mean",
        "spread": "fidelity_std",
        "label": "LIME local fit (exp.score, R²)",
        "title": "Local fidelity",
        "log_y": False,
    },
    "spearman": {
        "column": "spearman_stability_mean",
        "spread": None,
        "label": "Mean pairwise Spearman ρ",
        "title": "Stability: weight ranking",
        "log_y": False,
        "ylim": (-0.05, 1.05),
    },
    "jaccard": {
        "column": "top5_jaccard_mean",
        "spread": None,
        "label": "Mean pairwise top-5 Jaccard",
        "title": "Stability: top-5 features",
        "log_y": False,
        "ylim": (-0.05, 1.05),
    },
}


def _style_axis(ax: plt.Axes, budgets: Iterable[int], log_y: bool = False) -> None:
    budgets = sorted(set(int(b) for b in budgets))
    ax.set_xscale("log")
    ax.set_xticks(budgets, labels=[f"{b:,}" for b in budgets])
    ax.minorticks_off()
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _ordered_models(models: Iterable[str]) -> list[str]:
    models = list(dict.fromkeys(models))
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _model_colour(model: str) -> str:
    if model in MODEL_COLOURS:
        return MODEL_COLOURS[model]
    # Any extra model (e.g. a renamed run) gets the next fixed slot by name order.
    extra_index = sum(1 for _ in model) % len(EXTRA_COLOURS)
    return EXTRA_COLOURS[extra_index]


def _label_line_end(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Queue a direct label for the right end of a line; drawn by _place_labels."""
    if np.isfinite(y):
        ax.__dict__.setdefault("_end_labels", []).append((x, y, text))


def _place_labels(ax: plt.Axes, min_gap_px: float = 11.0) -> None:
    """Draw queued end-of-line labels, nudged apart vertically so they never overlap.

    Call after the figure layout is final. Text is in ink, not the series colour.
    """
    labels = ax.__dict__.pop("_end_labels", [])
    if not labels:
        return
    to_display = ax.transData.transform
    points = sorted(
        ((*to_display((x, y)), text) for x, y, text in labels), key=lambda p: p[1]
    )
    ys = [p[1] for p in points]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    # Re-centre the stack on the original positions so labels stay near their lines.
    shift = (np.mean([p[1] for p in points]) - np.mean(ys)) if len(ys) > 1 else 0.0
    inverse = ax.transData.inverted().transform
    for (x_px, _, text), y_px in zip(points, ys):
        x_data, y_data = inverse((x_px, y_px + shift))
        ax.annotate(text, (x_data, y_data), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK, annotation_clip=False)


def load_summaries(directory: str | Path = ".") -> pd.DataFrame:
    """Read every ``*_controlled_lime_summary.csv`` in ``directory`` into one frame."""
    paths = sorted(Path(directory).glob(f"*{SUMMARY_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"No *{SUMMARY_SUFFIX} files in {Path(directory).resolve()}. Run the "
            "controlled benchmark cell of each model notebook first (or copy the CSVs here)."
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def plot_model_results(
    summary: pd.DataFrame,
    model_name: str | None = None,
    out_dir: str | Path = "figures",
    show: bool = True,
) -> Path:
    """Runtime, fidelity and both stability measures vs LIME sample budget for one model.

    One line per LIME perturbation dimension (18 / 36 / 54 features). Error bars on
    runtime and fidelity are ±1 SD across all cases and repeats.
    """
    if model_name is None:
        model_name = str(summary["model"].iloc[0])
    data = summary[summary["model"] == model_name] if "model" in summary else summary
    if data.empty:
        raise ValueError(f"No rows for model {model_name!r} in the summary.")
    budgets = data["num_samples"].unique()
    feature_counts = sorted(data["feature_count"].unique())
    cases = int(data["cases"].max()) if "cases" in data else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, spec) in zip(axes.flat, METRICS.items()):
        for i, count in enumerate(feature_counts):
            rows = data[data["feature_count"] == count].sort_values("num_samples")
            colour = FEATURE_COUNT_COLOURS[i % len(FEATURE_COUNT_COLOURS)]
            y = rows[spec["column"]].to_numpy(dtype=float)
            x = rows["num_samples"].to_numpy(dtype=float)
            ax.plot(x, y, color=colour, linewidth=2, marker=FEATURE_COUNT_MARKERS[i % 3],
                    markersize=6, label=f"{count} features")
            if spec["spread"] and spec["spread"] in rows:
                spread = rows[spec["spread"]].to_numpy(dtype=float)
                lower = np.maximum(y - spread, y * 0.05) if spec["log_y"] else y - spread
                ax.fill_between(x, lower, y + spread, color=colour, alpha=0.08, linewidth=0)
            _label_line_end(ax, x[-1], y[-1], f"{count}")
        ax.set_title(spec["title"], fontsize=11, loc="left")
        ax.set_xlabel("LIME perturbation samples (num_samples)")
        ax.set_ylabel(spec["label"])
        if "ylim" in spec:
            ax.set_ylim(*spec["ylim"])
        _style_axis(ax, budgets, spec["log_y"])

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False,
               fontsize=9, title="LIME perturbation dimension", title_fontsize=9)
    subtitle = (f"Means over {cases} held-out cases × 5 repeats; shading on runtime and "
                "fidelity is ±1 SD." if cases else "")
    fig.suptitle(f"{model_name}: LIME cost and quality vs sample budget", x=0.02, y=0.995,
                 ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.962, subtitle, fontsize=9, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ax in axes.flat:
        _place_labels(ax)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name.lower()}_lime_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_model_comparison(
    summaries: pd.DataFrame | str | Path = ".",
    out_dir: str | Path = "figures",
    show: bool = True,
) -> list[Path]:
    """Compare every model on the same axes.

    Writes one figure per metric (one panel per perturbation dimension, one line per
    model) plus a cost-vs-stability trade-off figure. ``summaries`` is either a
    combined summary DataFrame or a folder containing the summary CSVs.
    """
    data = summaries if isinstance(summaries, pd.DataFrame) else load_summaries(summaries)
    models = _ordered_models(data["model"])
    feature_counts = sorted(data["feature_count"].unique())
    budgets = data["num_samples"].unique()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def model_lines(ax: plt.Axes, panel: pd.DataFrame, x_col: str, y_col: str,
                    direct_labels: bool = True) -> None:
        for model in models:
            rows = panel[panel["model"] == model].sort_values("num_samples")
            if rows.empty:
                continue
            x = rows[x_col].to_numpy(dtype=float)
            y = rows[y_col].to_numpy(dtype=float)
            ax.plot(x, y, color=_model_colour(model), linewidth=2,
                    marker=MODEL_MARKERS.get(model, "o"), markersize=6, label=model)
            if direct_labels:
                _label_line_end(ax, x[-1], y[-1], model)

    for key, spec in METRICS.items():
        fig, axes = plt.subplots(1, len(feature_counts), figsize=(5.5 * len(feature_counts), 4.4),
                                 sharey=True, squeeze=False)
        for ax, count in zip(axes[0], feature_counts):
            model_lines(ax, data[data["feature_count"] == count], "num_samples", spec["column"])
            ax.set_title(f"{count} perturbed features", fontsize=10, loc="left")
            ax.set_xlabel("LIME perturbation samples")
            if "ylim" in spec:
                ax.set_ylim(*spec["ylim"])
            _style_axis(ax, budgets, spec["log_y"])
            # Leave room on the right for the direct model labels.
            ax.set_xlim(min(budgets) / 1.4, max(budgets) * 4.5)
        axes[0][0].set_ylabel(spec["label"])
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
        fig.suptitle(f"{spec['title']} by model", x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for ax in axes[0]:
            _place_labels(ax)
        path = out_dir / f"comparison_{key}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        written.append(path)
        if show:
            plt.show()
        plt.close(fig)

    # Cost/quality trade-off: where does extra runtime stop buying stability?
    fig, axes = plt.subplots(1, len(feature_counts), figsize=(5 * len(feature_counts), 4.4),
                             sharey=True, squeeze=False)
    for ax, count in zip(axes[0], feature_counts):
        panel = data[data["feature_count"] == count]
        model_lines(ax, panel, "runtime_mean_seconds", "spearman_stability_mean",
                    direct_labels=False)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED_INK, labelsize=8)
        ax.set_title(f"{count} perturbed features", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("Seconds per explanation (log scale)")
        ax.margins(x=0.25)
    axes[0][0].set_ylabel("Mean pairwise Spearman ρ")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Cost vs stability (each point is a sample budget, 500 → 10,000)",
                 x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "comparison_cost_vs_stability.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    written.append(path)
    if show:
        plt.show()
    plt.close(fig)

    # Also keep one tidy table of every model's numbers next to the graphs.
    table_path = out_dir / "comparison_summary_table.csv"
    data.sort_values(["model", "feature_count", "num_samples"]).to_csv(table_path, index=False)
    written.append(table_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=".", help="folder containing *_controlled_lime_summary.csv")
    parser.add_argument("--out", default="figures", help="folder to write the PNGs into")
    args = parser.parse_args()

    data = load_summaries(args.dir)
    for model in _ordered_models(data["model"]):
        print("wrote", plot_model_results(data, model, args.out, show=False))
    for path in plot_model_comparison(data, args.out, show=False):
        print("wrote", path)


if __name__ == "__main__":
    main()
