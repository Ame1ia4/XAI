"""Shared data split and LIME benchmark protocol for the Covertype experiments."""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer


RANDOM_STATE = 42
SAMPLE_SIZE = 10_000
TEST_SIZE = 0.20
BACKGROUND_SIZE = 2_000
SAMPLE_BUDGETS = (500, 1_000, 2_500, 5_000, 10_000)
REPEATS = 5
EXPLANATIONS_PER_CLASS = 3
NUM_FEATURES = 10
TOP_K = 5
FEATURE_COUNTS = (18, 36, 54)
CLASS_NAMES = (
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
)


def load_controlled_covtype() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Return the same seeded, stratified 10k-row sample and split for every model."""
    dataset = fetch_covtype(as_frame=True)
    X_full = dataset.data
    y_full = dataset.target.astype(int) - 1

    X_sample, _, y_sample, _ = train_test_split(
        X_full,
        y_full,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sample,
    )
    return X_sample, y_sample, X_train, X_test, y_train, y_test


def selected_explanation_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Choose three held-out rows per class without conditioning on model predictions."""
    labels = pd.Series(y_test.to_numpy(), index=X_test.index, name="cover_type")
    cases = (
        X_test.assign(_cover_type=labels)
        .groupby("_cover_type", group_keys=False, sort=True)
        .sample(n=EXPLANATIONS_PER_CLASS, random_state=RANDOM_STATE)
    )
    return cases.drop(columns="_cover_type").sort_index()


def feature_subset(feature_count: int, feature_names: list[str]) -> list[int]:
    """Return a fixed nested subset: all numeric features plus seeded binary indicators."""
    numeric_indices = list(range(min(10, len(feature_names))))
    categorical_indices = list(range(len(numeric_indices), len(feature_names)))
    if feature_count < len(numeric_indices) or feature_count > len(feature_names):
        raise ValueError(
            f"feature_count must be between {len(numeric_indices)} and {len(feature_names)}"
        )
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_categorical = rng.permutation(categorical_indices).tolist()
    return sorted(numeric_indices + shuffled_categorical[: feature_count - len(numeric_indices)])


def _protocol_settings() -> dict[str, Any]:
    """Settings a checkpoint must share with the current run to be resumable."""
    return {
        "RANDOM_STATE": RANDOM_STATE,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "TEST_SIZE": TEST_SIZE,
        "BACKGROUND_SIZE": BACKGROUND_SIZE,
        "SAMPLE_BUDGETS": tuple(SAMPLE_BUDGETS),
        "REPEATS": REPEATS,
        "EXPLANATIONS_PER_CLASS": EXPLANATIONS_PER_CLASS,
        "NUM_FEATURES": NUM_FEATURES,
        "TOP_K": TOP_K,
        "FEATURE_COUNTS": tuple(FEATURE_COUNTS),
    }


def _load_checkpoint(path: Path, model_name: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "model": model_name,
            "settings": _protocol_settings(),
            "completed_units": [],
            "observations": [],
            "stability_rows": [],
        }
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if state.get("model") != model_name or state.get("settings") != _protocol_settings():
        raise ValueError(
            f"Checkpoint {path} was written for a different model or protocol "
            "configuration. Delete it (or pass a different checkpoint_path) to start over."
        )
    return state


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    # Write to a temporary file first so a crash mid-write cannot corrupt the checkpoint.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle)
    os.replace(temporary, path)


def benchmark_lime(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    model_name: str,
    checkpoint_path: str | Path | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark LIME cost, fit score, and repeated-run stability on shared cases.

    ``checkpoint_path`` and ``progress`` are optional and do not change the results.
    With a checkpoint path, finished (case, feature_count) units are saved as they
    complete and skipped on the next call, so a long run (e.g. TabFM on a GPU) can be
    interrupted and resumed. Every LIME explainer is seeded independently, so a
    resumed run produces the same explanations as an uninterrupted one.
    """
    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    )
    feature_names = X_train.columns.tolist()
    cases = selected_explanation_cases(X_test, y_test)
    class_positions = {
        class_label: position
        for position, class_label in enumerate(model.classes_)
    }

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    state = (
        _load_checkpoint(checkpoint, model_name)
        if checkpoint is not None
        else {"completed_units": [], "observations": [], "stability_rows": []}
    )
    completed_units = {tuple(unit) for unit in state["completed_units"]}
    observations: list[dict[str, Any]] = state["observations"]
    stability_rows: list[dict[str, Any]] = state["stability_rows"]

    total_units = len(cases) * len(FEATURE_COUNTS)
    if progress and completed_units:
        print(f"Resuming {model_name}: {len(completed_units)}/{total_units} units already done.", flush=True)

    # Exclude first-call overhead from the per-explanation timing.
    predict_proba(background.iloc[:1].to_numpy())

    run_started = perf_counter()
    units_run_now = 0
    for case_index, row in cases.iterrows():
        if all((case_index, count) in completed_units for count in FEATURE_COUNTS):
            continue
        case_data = row.to_numpy()
        true_label = int(y_test.loc[case_index])
        predicted_label = int(model.predict(X_test.loc[[case_index]])[0])
        label_position = class_positions[predicted_label]
        for feature_count in FEATURE_COUNTS:
            if (case_index, feature_count) in completed_units:
                continue
            unit_started = perf_counter()
            subset_indices = feature_subset(feature_count, feature_names)
            subset_names = [feature_names[index] for index in subset_indices]
            background_subset = background.iloc[:, subset_indices].to_numpy()
            categorical_positions = [
                position
                for position, original_index in enumerate(subset_indices)
                if original_index >= 10
            ]
            categorical_names = {
                index: ["0", "1"] for index in categorical_positions
            }

            def predict_in_subset(rows: np.ndarray) -> np.ndarray:
                reconstructed = np.tile(case_data, (len(rows), 1))
                reconstructed[:, subset_indices] = rows
                return predict_proba(reconstructed)

            unit_observations: list[dict[str, Any]] = []
            vectors_by_budget: dict[int, list[np.ndarray]] = {}
            for budget in SAMPLE_BUDGETS:
                vectors: list[np.ndarray] = []
                for repeat in range(REPEATS):
                    explainer = LimeTabularExplainer(
                        training_data=background_subset,
                        feature_names=subset_names,
                        class_names=list(CLASS_NAMES),
                        categorical_features=categorical_positions,
                        categorical_names=categorical_names,
                        mode="classification",
                        random_state=RANDOM_STATE + repeat,
                    )
                    started = perf_counter()
                    explanation = explainer.explain_instance(
                        data_row=case_data[subset_indices],
                        predict_fn=predict_in_subset,
                        labels=(label_position,),
                        num_features=min(NUM_FEATURES, feature_count),
                        num_samples=budget,
                    )
                    runtime_seconds = perf_counter() - started

                    weights = np.zeros(feature_count, dtype=float)
                    for feature_index, weight in explanation.as_map()[label_position]:
                        weights[feature_index] = weight
                    vectors.append(weights)
                    unit_observations.append(
                        {
                            "model": model_name,
                            "case_index": case_index,
                            "true_class": true_label,
                            "predicted_class": predicted_label,
                            "correct": true_label == predicted_label,
                            "feature_count": feature_count,
                            "num_samples": budget,
                            "repeat": repeat + 1,
                            "runtime_seconds": runtime_seconds,
                            "local_fidelity_score": float(explanation.score),
                        }
                    )
                vectors_by_budget[budget] = vectors

            unit_stability: list[dict[str, Any]] = []
            for budget, vectors in vectors_by_budget.items():
                for first, second in combinations(vectors, 2):
                    selected = np.flatnonzero((first != 0) | (second != 0))
                    correlation = (
                        spearmanr(first[selected], second[selected]).statistic
                        if len(selected) >= 2
                        else np.nan
                    )
                    first_selected = np.flatnonzero(first)
                    second_selected = np.flatnonzero(second)
                    first_top = set(
                        first_selected[
                            np.argsort(np.abs(first[first_selected]))[-TOP_K:]
                        ]
                    )
                    second_top = set(
                        second_selected[
                            np.argsort(np.abs(second[second_selected]))[-TOP_K:]
                        ]
                    )
                    union = first_top | second_top
                    unit_stability.append(
                        {
                            "model": model_name,
                            "case_index": case_index,
                            "feature_count": feature_count,
                            "num_samples": budget,
                            "spearman": correlation,
                            "top5_jaccard": (
                                len(first_top & second_top) / len(union)
                                if union
                                else np.nan
                            ),
                        }
                    )

            # Commit the whole unit at once so a checkpoint never holds half a unit.
            observations.extend(unit_observations)
            stability_rows.extend(unit_stability)
            completed_units.add((case_index, feature_count))
            state["completed_units"] = sorted(completed_units, key=str)
            if checkpoint is not None:
                _save_checkpoint(checkpoint, state)

            units_run_now += 1
            if progress:
                done = len(completed_units)
                elapsed = perf_counter() - run_started
                remaining = (total_units - done) * elapsed / units_run_now
                print(
                    f"[{model_name}] {done}/{total_units} units | case {case_index}, "
                    f"{feature_count} features: {perf_counter() - unit_started:.1f}s | "
                    f"est. remaining {remaining / 60:.1f} min",
                    flush=True,
                )

    raw = pd.DataFrame(observations)
    stability = pd.DataFrame(stability_rows)
    summary = (
        raw.groupby(["model", "feature_count", "num_samples"], as_index=False)
        .agg(
            cases=("case_index", "nunique"),
            explanation_runs=("repeat", "size"),
            correct_case_fraction=("correct", "mean"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
            runtime_median_seconds=("runtime_seconds", "median"),
            runtime_std_seconds=("runtime_seconds", "std"),
            fidelity_mean=("local_fidelity_score", "mean"),
            fidelity_std=("local_fidelity_score", "std"),
        )
        .merge(
            stability.groupby(
                ["model", "feature_count", "num_samples"], as_index=False
            ).agg(
                spearman_stability_mean=("spearman", "mean"),
                top5_jaccard_mean=("top5_jaccard", "mean"),
            ),
            on=["model", "feature_count", "num_samples"],
        )
    )
    return raw, summary


def evaluate_predictive_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return comparable overall and class-balanced test metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

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
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
    }
