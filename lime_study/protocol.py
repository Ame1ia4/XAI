"""Shared data split, LIME benchmark, metrics, and result saving for every model.

Ported from experiments/controlled_lime_protocol.py with identical sampling and
measurement logic; LIME settings are now passed explicitly from config.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
from math import comb
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from lime_study import config
from lime_study.threads import thread_env_snapshot


GROUP_COLUMNS = ["model", "feature_count", "num_samples"]
PACKAGES = ("lime", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib")


# --- Data -----------------------------------------------------------------
def load_controlled_covtype() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the same seeded, stratified 10k-row sample and 80/20 split for every model."""
    dataset = fetch_covtype(as_frame=True)
    X_full = dataset.data
    y_full = dataset.target.astype(int) - 1  # 1..7 -> 0..6

    X_sample, _, y_sample, _ = train_test_split(
        X_full,
        y_full,
        train_size=config.SAMPLE_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_full,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_sample,
    )
    return X_train, X_test, y_train, y_test


def selected_explanation_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    per_class: int,
) -> pd.DataFrame:
    """Choose held-out rows per true class without conditioning on model predictions."""
    labels = pd.Series(y_test.to_numpy(), index=X_test.index, name="cover_type")
    cases = (
        X_test.assign(_cover_type=labels)
        .groupby("_cover_type", group_keys=False, sort=True)
        .sample(n=per_class, random_state=config.RANDOM_STATE)
    )
    return cases.drop(columns="_cover_type").sort_index()


def feature_subset(feature_count: int, n_features: int) -> list[int]:
    """Return a fixed nested subset: all numeric features plus seeded binary indicators."""
    numeric_indices = list(range(min(config.N_NUMERIC, n_features)))
    categorical_indices = list(range(len(numeric_indices), n_features))
    if feature_count < len(numeric_indices) or feature_count > n_features:
        raise ValueError(
            f"feature_count must be between {len(numeric_indices)} and {n_features}"
        )
    rng = np.random.default_rng(config.RANDOM_STATE)
    shuffled_categorical = rng.permutation(categorical_indices).tolist()
    return sorted(numeric_indices + shuffled_categorical[: feature_count - len(numeric_indices)])


def make_predict_proba(
    model: Any, feature_names: list[str]
) -> Callable[[np.ndarray], np.ndarray]:
    """Shared wrapper: numpy rows -> named DataFrame -> model probabilities."""

    def predict_proba(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(rows, columns=feature_names))

    return predict_proba


# --- Metrics --------------------------------------------------------------
def evaluate_predictive_performance(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Return comparable overall and class-balanced test metrics."""
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_precision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }


def pairwise_stability(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Spearman over the union of selected features, and top-k Jaccard."""
    selected = np.flatnonzero((first != 0) | (second != 0))
    correlation = (
        float(spearmanr(first[selected], second[selected]).statistic)
        if len(selected) >= 2
        else np.nan
    )

    def top_k(weights: np.ndarray) -> set[int]:
        nonzero = np.flatnonzero(weights)
        return set(nonzero[np.argsort(np.abs(weights[nonzero]))[-config.TOP_K :]])

    first_top, second_top = top_k(first), top_k(second)
    union = first_top | second_top
    jaccard = len(first_top & second_top) / len(union) if union else np.nan
    return correlation, jaccard


# --- Benchmark ------------------------------------------------------------
def benchmark_lime(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    sweep: config.Sweep,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark LIME cost, fit score, and repeated-run stability on shared cases."""
    feature_names = X_train.columns.tolist()
    predict_proba = make_predict_proba(model, feature_names)
    background = X_train.sample(
        n=min(config.BACKGROUND_SIZE, len(X_train)),
        random_state=config.RANDOM_STATE,
    )
    cases = selected_explanation_cases(X_test, y_test, sweep.explanations_per_class)
    class_positions = {label: position for position, label in enumerate(model.classes_)}

    # Exclude first-call overhead from the per-explanation timing.
    predict_proba(background.iloc[:1].to_numpy())

    observations: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for case_number, (case_index, row) in enumerate(cases.iterrows(), start=1):
        case_started = perf_counter()
        case_data = row.to_numpy()
        true_label = int(y_test.loc[case_index])
        predicted_label = int(model.predict(X_test.loc[[case_index]])[0])
        label_position = class_positions[predicted_label]
        for feature_count in config.FEATURE_COUNTS:
            subset_indices = feature_subset(feature_count, len(feature_names))
            subset_names = [feature_names[index] for index in subset_indices]
            background_subset = background.iloc[:, subset_indices].to_numpy()
            categorical_positions = [
                position
                for position, original_index in enumerate(subset_indices)
                if original_index >= config.N_NUMERIC
            ]
            categorical_names = {index: ["0", "1"] for index in categorical_positions}

            def predict_in_subset(rows: np.ndarray) -> np.ndarray:
                # Features outside the subset stay at the case's observed values.
                reconstructed = np.tile(case_data, (len(rows), 1))
                reconstructed[:, subset_indices] = rows
                return predict_proba(reconstructed)

            for budget in sweep.sample_budgets:
                vectors: list[np.ndarray] = []
                for repeat in range(sweep.repeats):
                    seed = config.RANDOM_STATE + repeat
                    explainer = LimeTabularExplainer(
                        training_data=background_subset,
                        feature_names=subset_names,
                        class_names=list(config.CLASS_NAMES),
                        categorical_features=categorical_positions,
                        categorical_names=categorical_names,
                        random_state=seed,
                        **config.LIME_EXPLAINER_KWARGS,
                    )
                    started = perf_counter()
                    explanation = explainer.explain_instance(
                        data_row=case_data[subset_indices],
                        predict_fn=predict_in_subset,
                        labels=(label_position,),
                        num_features=min(config.NUM_FEATURES, feature_count),
                        num_samples=budget,
                        distance_metric=config.LIME_DISTANCE_METRIC,
                        model_regressor=config.make_surrogate(seed),
                    )
                    runtime_seconds = perf_counter() - started

                    weights = np.zeros(feature_count, dtype=float)
                    for feature_index, weight in explanation.as_map()[label_position]:
                        weights[feature_index] = weight
                    vectors.append(weights)
                    observations.append(
                        {
                            "model": model_name,
                            "case_index": case_index,
                            "true_class": true_label,
                            "predicted_class": predicted_label,
                            "correct": true_label == predicted_label,
                            "feature_count": feature_count,
                            "num_samples": budget,
                            "repeat": repeat + 1,
                            "seed": seed,
                            "runtime_seconds": runtime_seconds,
                            "local_fidelity_score": float(explanation.score),
                            "weights_json": json.dumps(
                                {
                                    subset_names[i]: float(weights[i])
                                    for i in np.flatnonzero(weights)
                                }
                            ),
                        }
                    )

                for first, second in combinations(vectors, 2):
                    spearman, jaccard = pairwise_stability(first, second)
                    stability_rows.append(
                        {
                            "model": model_name,
                            "case_index": case_index,
                            "feature_count": feature_count,
                            "num_samples": budget,
                            "spearman": spearman,
                            f"top{config.TOP_K}_jaccard": jaccard,
                        }
                    )
        print(
            f"[{model_name}] case {case_number}/{len(cases)} done "
            f"({perf_counter() - case_started:.1f}s)",
            flush=True,
        )

    return pd.DataFrame(observations), pd.DataFrame(stability_rows)


def summarise(raw: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per model x feature count x budget."""
    jaccard = f"top{config.TOP_K}_jaccard"
    explanation_summary = raw.groupby(GROUP_COLUMNS, as_index=False).agg(
        cases=("case_index", "nunique"),
        explanation_runs=("repeat", "size"),
        correct_case_fraction=("correct", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
        runtime_median_seconds=("runtime_seconds", "median"),
        runtime_std_seconds=("runtime_seconds", "std"),
        fidelity_mean=("local_fidelity_score", "mean"),
        fidelity_median=("local_fidelity_score", "median"),
        fidelity_std=("local_fidelity_score", "std"),
    )
    stability_summary = stability.groupby(GROUP_COLUMNS, as_index=False).agg(
        stability_pairs=("spearman", "size"),
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        spearman_nan=("spearman", lambda s: int(s.isna().sum())),
        jaccard_mean=(jaccard, "mean"),
        jaccard_std=(jaccard, "std"),
        jaccard_nan=(jaccard, lambda s: int(s.isna().sum())),
    )
    return explanation_summary.merge(stability_summary, on=GROUP_COLUMNS)


def validate(
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    summary: pd.DataFrame,
    sweep: config.Sweep,
) -> None:
    """Fail loudly if the run does not have the expected shape."""
    n_cases = len(config.CLASS_NAMES) * sweep.explanations_per_class
    n_groups = len(config.FEATURE_COUNTS) * len(sweep.sample_budgets)
    expected = {
        "raw rows": (len(raw), n_cases * n_groups * sweep.repeats),
        "stability rows": (len(stability), n_cases * n_groups * comb(sweep.repeats, 2)),
        "summary rows": (len(summary), n_groups),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise AssertionError(f"Expected {wanted} {label}, got {actual}")
    if not (summary["cases"] == n_cases).all():
        raise AssertionError(f"Every group should contain {n_cases} cases")
    if not (summary["explanation_runs"] == n_cases * sweep.repeats).all():
        raise AssertionError(f"Every group should contain {n_cases * sweep.repeats} runs")
    nan_total = int(summary["spearman_nan"].sum() + summary["jaccard_nan"].sum())
    if nan_total:
        print(f"WARNING: {nan_total} NaN stability values; inspect *_stability.csv", flush=True)


# --- Saving ---------------------------------------------------------------
def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_results(
    out_dir: Path,
    model_name: str,
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    summary: pd.DataFrame,
    performance: dict[str, float],
    run_meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / f"{model_name}_raw.csv", index=False)
    stability.to_csv(out_dir / f"{model_name}_stability.csv", index=False)
    summary.to_csv(out_dir / f"{model_name}_summary.csv", index=False)
    (out_dir / f"{model_name}_performance.json").write_text(json.dumps(performance, indent=2))
    (out_dir / f"{model_name}_run_meta.json").write_text(
        json.dumps(run_meta, indent=2, default=str)
    )


# --- Driver ---------------------------------------------------------------
def run_model_experiment(model_name: str, quick: bool = False) -> pd.DataFrame:
    """Fit one model, benchmark LIME on it, validate, and save every output."""
    sweep = config.QUICK_SWEEP if quick else config.FULL_SWEEP
    out_dir = config.results_dir(sweep)
    print(f"[{model_name}] sweep={sweep.name} -> {out_dir}", flush=True)

    X_train, X_test, y_train, y_test = load_controlled_covtype()
    model = config.MODEL_BUILDERS[model_name]()
    fit_started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - fit_started
    performance = evaluate_predictive_performance(model, X_test, y_test)
    print(f"[{model_name}] fitted in {fit_seconds:.1f}s; test metrics {performance}", flush=True)

    benchmark_started = perf_counter()
    raw, stability = benchmark_lime(model, X_train, X_test, y_test, model_name, sweep)
    benchmark_seconds = perf_counter() - benchmark_started
    summary = summarise(raw, stability)
    validate(raw, stability, summary, sweep)

    run_meta = {
        "model": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "thread_env": thread_env_snapshot(),
        "config": config.snapshot(sweep),
        "model_params": model.get_params(deep=True),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "fit_seconds": fit_seconds,
        "benchmark_seconds": benchmark_seconds,
    }
    save_results(out_dir, model_name, raw, stability, summary, performance, run_meta)
    print(f"[{model_name}] done in {benchmark_seconds:.1f}s; results in {out_dir}", flush=True)
    return summary


def cli(model_name: str) -> None:
    """Entry point shared by the per-model scripts."""
    parser = argparse.ArgumentParser(description=f"Controlled LIME benchmark: {model_name}")
    parser.add_argument(
        "--quick", action="store_true", help="tiny smoke-test sweep written to results/quick"
    )
    args = parser.parse_args()
    run_model_experiment(model_name, quick=args.quick)
