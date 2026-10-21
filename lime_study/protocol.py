"""Shared data split, LIME benchmark, metrics, and result saving for every model.

The dataset is UCI HAR (561 features) and LIME settings come explicitly from config.
All models run in one process as a randomized complete block design, every model
explains the case's true class, and quality is scored afterwards, untimed, on held-out
perturbations and against large-N reference explanations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import re
import shutil
import ssl
import sys
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
from math import comb
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import certifi
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    pairwise_distances,
    precision_score,
    r2_score,
    recall_score,
)

from lime_study import analysis, config
from lime_study.threads import thread_env_snapshot


GROUP_COLUMNS = analysis.GROUP_COLUMNS
PACKAGES = ("lime", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib")
REQUIREMENTS_FILE = config.PACKAGE_DIR / "requirements-pinned.txt"


# --- Data -----------------------------------------------------------------
INNER_ZIP = "UCI HAR Dataset.zip"
HAR_ROOT = "UCI HAR Dataset"


def _download_har() -> Path:
    """Download the UCI HAR archive once into config.DATA_DIR and return its path."""
    archive = config.DATA_DIR / "human_activity_recognition_using_smartphones.zip"
    if not archive.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading UCI HAR from {config.DATA_URL} ...", flush=True)
        partial = archive.with_suffix(".part")
        # certifi's CA bundle: python.org macOS builds ship without system root certificates.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(config.DATA_URL, context=context) as response, open(
            partial, "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        partial.rename(archive)
    return archive


def _unique_feature_names(names: list[str]) -> list[str]:
    """features.txt repeats 42 bandsEnergy names three times; suffix repeats as __2, __3."""
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return unique


def load_har() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]
]:
    """Return the official subject-based UCI HAR split, test-row participants, and provenance.

    Train: 21 volunteers / 7,352 windows. Test: 9 other volunteers / 2,947 windows.
    Labels are shifted from 1..6 to 0..5 to match model class positions.
    """
    archive = _download_har()
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))

    def read_table(member: str, **kwargs: Any) -> pd.DataFrame:
        with inner.open(f"{HAR_ROOT}/{member}") as handle:
            return pd.read_csv(handle, sep=r"\s+", header=None, **kwargs)

    feature_names = _unique_feature_names(read_table("features.txt")[1].tolist())
    activity_labels = read_table("activity_labels.txt")[1].tolist()
    if tuple(activity_labels) != config.CLASS_NAMES:
        raise AssertionError(f"Unexpected activity labels: {activity_labels}")

    def read_split(split: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        X = read_table(f"{split}/X_{split}.txt", dtype=np.float64)
        X.columns = feature_names
        y = read_table(f"{split}/y_{split}.txt")[0].astype(int) - 1
        subjects = read_table(f"{split}/subject_{split}.txt")[0].to_numpy()
        return X, y.rename("activity"), subjects

    X_train, y_train, train_subjects = read_split("train")
    X_test, y_test, test_subjects = read_split("test")
    # Give the test rows distinct indices so case indices never collide with training rows.
    X_test.index = y_test.index = pd.RangeIndex(len(X_train), len(X_train) + len(X_test))

    expected = {
        "X_train": (X_train.shape, (config.N_TRAIN_ROWS, config.N_FEATURES)),
        "X_test": (X_test.shape, (config.N_TEST_ROWS, config.N_FEATURES)),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise AssertionError(f"{label} has shape {actual}, expected {wanted}")
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise AssertionError(f"Subjects appear in both train and test: {sorted(overlap)}")

    provenance = {
        "data_url": config.DATA_URL,
        "data_doi": config.DATA_DOI,
        "archive_sha256": archive_sha256,
        "train_subjects": sorted(int(s) for s in set(train_subjects)),
        "test_subjects": sorted(int(s) for s in set(test_subjects)),
    }
    subject_test = pd.Series(test_subjects, index=X_test.index, name="subject")
    return X_train, X_test, y_train, y_test, subject_test, provenance


def selected_explanation_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    subjects: pd.Series,
    per_class: int,
) -> pd.DataFrame:
    """Choose held-out rows per true class, spread across participants, without model predictions.

    Rows from one participant may be correlated, so picks go round the classes in turn,
    and each pick is the row (in a seeded random order) whose participant has supplied
    the fewest cases so far. No participant may supply more than MAX_CASES_PER_SUBJECT.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    labels = pd.Series(y_test.to_numpy(), index=X_test.index)
    candidates = {
        label: list(rng.permutation(rows.index.to_numpy()))
        for label, rows in labels.groupby(labels, sort=True)
    }
    used: dict[int, int] = {}
    chosen = []
    for _ in range(per_class):
        for label, rows in candidates.items():
            pick = min(rows, key=lambda row: used.get(subjects.loc[row], 0))  # first in random order on ties
            rows.remove(pick)
            chosen.append(pick)
            used[subjects.loc[pick]] = used.get(subjects.loc[pick], 0) + 1
    if max(used.values()) > config.MAX_CASES_PER_SUBJECT:
        raise AssertionError(f"A participant supplies {max(used.values())} cases; cap is {config.MAX_CASES_PER_SUBJECT}")
    return X_test.loc[sorted(chosen)]


def model_fingerprint(model: Any, X_test: pd.DataFrame) -> str:
    """SHA-256 of the model's rounded test-set probabilities.

    Identical fingerprints across runs (variants) mean the same fitted model was explained,
    which plot_results checks before results from different variants are set side by side.
    """
    probabilities = np.ascontiguousarray(np.round(model.predict_proba(X_test), 10))
    return hashlib.sha256(probabilities.tobytes()).hexdigest()


def feature_ranking(variant: config.Variant, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """One fixed order of all features; every feature count takes a prefix of it.

    "random": a seeded permutation. "ranked": descending ANOVA F against the training
    labels, which is model-agnostic, so every model still gets the same subsets.
    Taking prefixes of one order makes 100 ⊂ 300 ⊂ 500 in both cases.
    """
    if variant.subset == "random":
        return np.random.default_rng(config.RANDOM_STATE).permutation(X_train.shape[1])
    if variant.subset == "ranked":
        scores, _ = f_classif(X_train, y_train)
        return np.argsort(-np.nan_to_num(scores, nan=0.0), kind="stable")
    raise ValueError(f"Unknown subset strategy {variant.subset!r}")


def feature_subset(feature_count: int, ranking: np.ndarray) -> list[int]:
    if not 1 <= feature_count <= len(ranking):
        raise ValueError(f"feature_count must be between 1 and {len(ranking)}")
    return sorted(ranking[:feature_count].tolist())


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


def top_k(weights: np.ndarray) -> set[int]:
    nonzero = np.flatnonzero(weights)
    return set(nonzero[np.argsort(np.abs(weights[nonzero]))[-config.TOP_K :]])


def pairwise_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Top-k Jaccard: shared top-k features (by |weight|) over their union.

    Spearman was dropped (20-10-2026): zero-filling unselected features makes its
    chance level depend on each model's weight signs, so it is not comparable across models.
    """
    first_top, second_top = top_k(first), top_k(second)
    union = first_top | second_top
    return len(first_top & second_top) / len(union) if union else np.nan


def holdout_draw(explainer: LimeTabularExplainer, data_row: np.ndarray) -> dict[str, np.ndarray]:
    """Fresh perturbations drawn and weighted exactly as explain_instance does it.

    Mirrors LimeTabularExplainer.explain_instance in lime==0.2.0.1 (sampling, scaling,
    distances, kernel). Sampling happens before any model is called, so with a fixed
    seed every model is scored on the same perturbations. Row 0 is the instance itself;
    it is kept apart for a sanity check.
    """
    if metadata.version("lime") != "0.2.0.1":
        raise RuntimeError("holdout_draw mirrors lime 0.2.0.1 internals; re-check before upgrading")
    data, inverse = explainer._LimeTabularExplainer__data_inverse(
        data_row, config.HOLDOUT_SAMPLES + 1
    )
    scaled = (data - explainer.scaler.mean_) / explainer.scaler.scale_
    distances = pairwise_distances(
        scaled, scaled[0].reshape(1, -1), metric=config.LIME_DISTANCE_METRIC
    ).ravel()
    return {
        "instance": scaled[0],
        "scaled": scaled[1:],
        "inverse": inverse[1:],
        "weight": explainer.base.kernel_fn(distances[1:]),
    }


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def surrogate_predict(intercept: float, local_map: list[tuple[int, float]], scaled: np.ndarray) -> np.ndarray:
    prediction = np.full(scaled.shape[0], intercept, dtype=float)
    for feature_index, coefficient in local_map:
        prediction += coefficient * scaled[:, feature_index]
    return prediction


def heldout_scores(fit: dict[str, Any], holdout: dict[str, np.ndarray], target: np.ndarray) -> dict[str, float]:
    """Kernel-weighted R² and RMSE of the fitted surrogate on held-out perturbations.

    RMSE is LIME's own locality-weighted square loss (in probability units), which stays
    defined when the model's output barely moves; R² is relative to that movement.
    """
    at_instance = surrogate_predict(fit["intercept"], fit["local_map"], holdout["instance"][None, :])
    if not np.allclose(at_instance, fit["local_pred"], rtol=1e-6, atol=1e-9):
        raise AssertionError("Surrogate reconstruction does not match LIME's local prediction")
    prediction = surrogate_predict(fit["intercept"], fit["local_map"], holdout["scaled"])
    weights = holdout["weight"]
    return {
        "fidelity_heldout": float(r2_score(target, prediction, sample_weight=weights)),
        "heldout_rmse": float(np.sqrt(np.average((target - prediction) ** 2, weights=weights))),
    }


def speed_probe(repeats: int = 5) -> float:
    """Median seconds of a fixed numerical workload; repeated through the run to trace drift."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 300))
    y = X @ rng.normal(size=300)
    times = []
    for _ in range(repeats + 1):  # the first round warms BLAS and is discarded
        started = perf_counter()
        Ridge(alpha=1.0).fit(X, y)
        pairwise_distances(X, X[:1])
        times.append(perf_counter() - started)
    return float(np.median(times[1:]))


# --- Benchmark ------------------------------------------------------------
def benchmark_lime(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    subject_test: pd.Series,
    sweep: config.Sweep,
    variant: config.Variant,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, float]]]:
    """Benchmark LIME cost and quality for every model in one interleaved run.

    Phase 1 (timed): a randomized complete block design. Each block is one case x
    feature count x budget x repeat; blocks run in a seeded random order, and within a
    block every model runs once, in a fresh random order. Machine drift therefore hits
    all models alike, and paired runtime differences come from runs seconds apart.
    Phase 2 (untimed): held-out error and agreement with large-N references.
    Every model explains the case's true class, so all models explain the same thing.
    """
    names = list(models)
    feature_names = X_train.columns.tolist()
    predict_fns = {name: make_predict_proba(model, feature_names) for name, model in models.items()}
    background = X_train.sample(
        n=min(config.BACKGROUND_SIZE, len(X_train)),
        random_state=config.RANDOM_STATE,
    )
    cases = selected_explanation_cases(X_test, y_test, subject_test, sweep.explanations_per_class)
    ranking = feature_ranking(variant, X_train, y_train)
    lime_kwargs = config.lime_explainer_kwargs(variant)

    case_info: dict[int, dict[str, Any]] = {}
    for case_index, row in cases.iterrows():
        true_class = int(y_test.loc[case_index])
        case_info[case_index] = {
            "data": row.to_numpy(),
            "true_class": true_class,
            "subject": int(subject_test.loc[case_index]),
            "predicted_class": {
                name: int(model.predict(X_test.loc[[case_index]])[0]) for name, model in models.items()
            },
            "label_position": {
                name: list(model.classes_).index(true_class) for name, model in models.items()
            },
        }
    subsets = {}
    for feature_count in config.FEATURE_COUNTS:
        indices = feature_subset(feature_count, ranking)
        subsets[feature_count] = {
            "indices": indices,
            "names": [feature_names[i] for i in indices],
            "background": background.iloc[:, indices].to_numpy(),
        }

    def make_explainer(feature_count: int, seed: int) -> LimeTabularExplainer:
        return LimeTabularExplainer(
            training_data=subsets[feature_count]["background"],
            feature_names=subsets[feature_count]["names"],
            class_names=list(config.CLASS_NAMES),
            random_state=seed,
            **lime_kwargs,
        )

    def subset_predict(name: str, case_index: int, feature_count: int) -> Callable[[np.ndarray], np.ndarray]:
        case_data = case_info[case_index]["data"]
        indices = subsets[feature_count]["indices"]
        predict_proba = predict_fns[name]

        def predict_in_subset(rows: np.ndarray) -> np.ndarray:
            # Features outside the subset stay at the case's observed values.
            reconstructed = np.tile(case_data, (len(rows), 1))
            reconstructed[:, indices] = rows
            return predict_proba(reconstructed)

        return predict_in_subset

    def explain(name: str, case_index: int, feature_count: int, num_samples: int, seed: int):
        label = case_info[case_index]["label_position"][name]
        explainer = make_explainer(feature_count, seed)
        predict_fn = subset_predict(name, case_index, feature_count)
        data_row = case_info[case_index]["data"][subsets[feature_count]["indices"]]
        # No automatic garbage collection inside the timed call (collected between calls instead).
        gc.disable()
        try:
            started = perf_counter()
            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                labels=(label,),
                num_features=min(config.NUM_FEATURES, feature_count),
                num_samples=num_samples,
                distance_metric=config.LIME_DISTANCE_METRIC,
                model_regressor=config.make_surrogate(seed),
            )
            runtime_seconds = perf_counter() - started
        finally:
            gc.enable()
        local_map = list(explanation.as_map()[label])
        weights = np.zeros(feature_count, dtype=float)
        for feature_index, weight in local_map:
            weights[feature_index] = weight
        fit = {
            "intercept": float(explanation.intercept[label]),
            "local_map": local_map,
            "local_pred": float(np.ravel(explanation.local_pred)[0]),
            "score": float(explanation.score),
            "weights": weights,
        }
        return fit, runtime_seconds

    def weights_json(feature_count: int, weights: np.ndarray) -> str:
        names_ = subsets[feature_count]["names"]
        return json.dumps({names_[i]: float(weights[i]) for i in np.flatnonzero(weights)})

    # Untimed warm-up per model x feature count: first-call overhead in LIME, models and numpy.
    warmup_case = next(iter(case_info))
    for name in names:
        predict_fns[name](background.iloc[:1].to_numpy())
        for feature_count in config.FEATURE_COUNTS:
            explain(name, warmup_case, feature_count, min(sweep.sample_budgets), config.HOLDOUT_SEED)

    # --- Phase 1: timed explanations, randomized complete blocks ------------
    blocks = [
        (case_index, feature_count, budget, repeat)
        for case_index in case_info
        for feature_count in config.FEATURE_COUNTS
        for budget in sweep.sample_budgets
        for repeat in range(sweep.repeats)
    ]
    rng = np.random.default_rng(config.RUN_ORDER_SEED)
    schedule = [
        (blocks[b], names[m], block_position)
        for block_position, b in enumerate(rng.permutation(len(blocks)), start=1)
        for m in rng.permutation(len(names))
    ]
    probe_every = max(1, len(schedule) // config.SPEED_PROBES)
    for _ in range(config.SPEED_PROBE_WARMUPS):  # unrecorded, so the first probe is not a cold start
        speed_probe()
    probes = [{"run_position": 0, "seconds": speed_probe()}]
    fits: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    observations: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    phase_started = perf_counter()
    for run_position, ((case_index, feature_count, budget, repeat), name, block_position) in enumerate(
        schedule, start=1
    ):
        seed = config.case_seed(config.RANDOM_STATE + repeat, case_index)
        fit, runtime_seconds = explain(name, case_index, feature_count, budget, seed)
        fits[(name, case_index, feature_count, budget, repeat)] = fit
        if run_position % 100 == 0:
            gc.collect()  # untimed; keeps memory bounded while timed calls run with GC off
        info = case_info[case_index]
        observations[name].append(
            {
                "model": name,
                "case_index": case_index,
                "subject": info["subject"],
                "true_class": info["true_class"],
                "predicted_class": info["predicted_class"][name],
                "correct": info["true_class"] == info["predicted_class"][name],
                "explained_class": info["true_class"],
                "feature_count": feature_count,
                "num_samples": budget,
                "repeat": repeat + 1,
                "seed": seed,
                "run_position": run_position,
                "block_position": block_position,
                "runtime_seconds": runtime_seconds,
                "fidelity_insample": fit["score"],
                "intercept": fit["intercept"],
                "weights_json": weights_json(feature_count, fit["weights"]),
            }
        )
        if run_position % probe_every == 0 or run_position == len(schedule):
            probes.append({"run_position": run_position, "seconds": speed_probe()})
            print(
                f"timed {run_position}/{len(schedule)} ({perf_counter() - phase_started:.0f}s)",
                flush=True,
            )

    # --- Phase 2 (untimed): held-out error and reference agreement ----------
    references: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    scores: dict[tuple[str, int, int, int, int], dict[str, float]] = {}
    for case_number, case_index in enumerate(case_info, start=1):
        info = case_info[case_index]
        for feature_count in config.FEATURE_COUNTS:
            data_row = info["data"][subsets[feature_count]["indices"]]
            holdout = holdout_draw(
                make_explainer(feature_count, config.case_seed(config.HOLDOUT_SEED, case_index)), data_row
            )
            for name in names:
                label = info["label_position"][name]
                target = subset_predict(name, case_index, feature_count)(holdout["inverse"])[:, label]
                target_sd = weighted_sd(target, holdout["weight"])
                flat = target_sd < config.FLAT_OUTPUT_SD
                reference_weights = []
                for reference_seed in (config.case_seed(s, case_index) for s in config.REFERENCE_SEEDS):
                    reference, reference_runtime = explain(
                        name, case_index, feature_count, sweep.reference_samples, reference_seed
                    )
                    reference_weights.append(reference["weights"])
                    references[name].append(
                        {
                            "model": name,
                            "case_index": case_index,
                            "explained_class": info["true_class"],
                            "feature_count": feature_count,
                            "num_samples": sweep.reference_samples,
                            "seed": reference_seed,
                            "runtime_seconds": reference_runtime,
                            "fidelity_insample": reference["score"],
                            **heldout_scores(reference, holdout, target),
                            "target_sd": target_sd,
                            "flat_output": flat,
                            "weights_json": weights_json(feature_count, reference["weights"]),
                        }
                    )
                ceiling = pairwise_jaccard(*reference_weights)
                for row in references[name][-len(reference_weights) :]:
                    row["reference_self_jaccard"] = ceiling
                for budget in sweep.sample_budgets:
                    for repeat in range(sweep.repeats):
                        key = (name, case_index, feature_count, budget, repeat)
                        scores[key] = {
                            **heldout_scores(fits[key], holdout, target),
                            "target_sd": target_sd,
                            "flat_output": flat,
                            "reference_jaccard": float(
                                np.nanmean(
                                    [pairwise_jaccard(fits[key]["weights"], r) for r in reference_weights]
                                )
                            ),
                            "reference_ceiling": ceiling,
                        }
        print(f"references {case_number}/{len(case_info)} done", flush=True)

    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for name in names:
        raw = pd.DataFrame(observations[name]).sort_values(
            ["case_index", "feature_count", "num_samples", "repeat"], ignore_index=True
        )
        keys = zip(raw["case_index"], raw["feature_count"], raw["num_samples"], raw["repeat"] - 1)
        score_frame = pd.DataFrame([scores[(name, *key)] for key in keys])
        raw = pd.concat([raw, score_frame], axis=1)

        stability_rows = []
        for case_index in case_info:
            for feature_count in config.FEATURE_COUNTS:
                for budget in sweep.sample_budgets:
                    vectors = [
                        fits[(name, case_index, feature_count, budget, repeat)]["weights"]
                        for repeat in range(sweep.repeats)
                    ]
                    for first, second in combinations(vectors, 2):
                        stability_rows.append(
                            {
                                "model": name,
                                "case_index": case_index,
                                "feature_count": feature_count,
                                "num_samples": budget,
                                f"top{config.TOP_K}_jaccard": pairwise_jaccard(first, second),
                            }
                        )
        tables[name] = {
            "raw": raw,
            "stability": pd.DataFrame(stability_rows),
            "references": pd.DataFrame(references[name]),
        }
    return tables, probes


def summarise(raw: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Per model x feature count x budget: case-level means with bootstrap CIs."""
    jaccard = f"top{config.TOP_K}_jaccard"
    counts = raw.groupby(GROUP_COLUMNS, as_index=False).agg(
        explanation_runs=("repeat", "size")
    ).merge(
        stability.groupby(GROUP_COLUMNS, as_index=False).agg(
            stability_pairs=(jaccard, "size"),
            jaccard_nan=(jaccard, lambda s: int(s.isna().sum())),
        ),
        on=GROUP_COLUMNS,
    )
    summary = analysis.summarise(analysis.case_level(raw, stability))
    return summary.merge(counts, on=GROUP_COLUMNS, validate="one_to_one")


def validate(
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    references: pd.DataFrame,
    summary: pd.DataFrame,
    sweep: config.Sweep,
) -> None:
    """Fail loudly if one model's results do not have the expected shape."""
    n_cases = len(config.CLASS_NAMES) * sweep.explanations_per_class
    n_groups = len(config.FEATURE_COUNTS) * len(sweep.sample_budgets)
    expected = {
        "raw rows": (len(raw), n_cases * n_groups * sweep.repeats),
        "stability rows": (len(stability), n_cases * n_groups * comb(sweep.repeats, 2)),
        "reference rows": (
            len(references),
            n_cases * len(config.FEATURE_COUNTS) * len(config.REFERENCE_SEEDS),
        ),
        "summary rows": (len(summary), n_groups),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise AssertionError(f"Expected {wanted} {label}, got {actual}")
    per_subject = raw.groupby("subject")["case_index"].nunique()
    if per_subject.max() > config.MAX_CASES_PER_SUBJECT:
        raise AssertionError(f"A participant supplies {per_subject.max()} cases; cap is {config.MAX_CASES_PER_SUBJECT}")
    if not (raw["explained_class"] == raw["true_class"]).all():
        raise AssertionError("Every model must explain the case's true class")
    if not (summary["cases"] == n_cases).all():
        raise AssertionError(f"Every group should contain {n_cases} cases")
    if not (summary["explanation_runs"] == n_cases * sweep.repeats).all():
        raise AssertionError(f"Every group should contain {n_cases * sweep.repeats} runs")
    if raw[["fidelity_heldout", "heldout_rmse", "target_sd"]].isna().any().any():
        raise AssertionError("Held-out scores are missing for some explanations")
    nan_total = int(summary["jaccard_nan"].sum()) + int(raw["reference_jaccard"].isna().sum())
    if nan_total:
        print(f"WARNING: {nan_total} NaN Jaccard values; inspect the CSVs", flush=True)


# --- Saving ---------------------------------------------------------------
def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def check_versions(strict: bool) -> dict[str, Any]:
    """Compare installed packages (and the Python minor version) with requirements-pinned.txt.

    Library versions change numerics and speed (e.g. BLAS, tree inference, LIME internals
    mirrored in holdout_draw), so paper runs must use the pinned environment. Strict mode
    stops before any work; otherwise mismatches are printed. Either way they are recorded.
    """
    text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    pinned = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)\s*$", text, flags=re.MULTILINE))
    installed: dict[str, str | None] = {}
    mismatches = []
    for package, wanted in pinned.items():
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = None
        if installed[package] != wanted:
            mismatches.append(f"{package}: installed {installed[package]}, pinned {wanted}")
    python_pin = re.search(r"Python (\d+\.\d+)", text)
    running_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_pin and python_pin.group(1) != running_python:
        mismatches.append(f"python: running {running_python}, pinned {python_pin.group(1)}")
    if mismatches:
        message = (
            "Environment does not match requirements-pinned.txt:\n  "
            + "\n  ".join(mismatches)
            + "\nInstall it with: pip install -r lime_study/requirements-pinned.txt"
        )
        if strict:
            raise RuntimeError(message + "\n(or pass --allow-version-mismatch for a non-paper run)")
        print(f"WARNING: {message}", flush=True)
    return {"pinned": pinned, "installed": installed, "mismatches": mismatches, "strict": strict}


def save_results(
    out_dir: Path,
    model_name: str,
    tables: dict[str, pd.DataFrame],
    performance: dict[str, float],
    run_meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{model_name}_{name}.csv", index=False)
    (out_dir / f"{model_name}_performance.json").write_text(json.dumps(performance, indent=2))
    (out_dir / f"{model_name}_run_meta.json").write_text(
        json.dumps(run_meta, indent=2, default=str)
    )


# --- Driver ---------------------------------------------------------------
def run_experiment(
    model_names: list[str],
    quick: bool = False,
    variant_name: str = config.MAIN_VARIANT.name,
    allow_version_mismatch: bool = False,
) -> None:
    """Fit the models, benchmark LIME on all of them in one interleaved run, save outputs.

    Only results from one run (one run_id) may be compared; plot_results enforces this.
    Full runs stop on any version mismatch unless explicitly allowed; quick runs warn.
    """
    version_check = check_versions(strict=not quick and not allow_version_mismatch)
    sweep = config.QUICK_SWEEP if quick else config.FULL_SWEEP
    variant = config.VARIANTS[variant_name]
    out_dir = config.results_dir(sweep, variant)
    run_id = uuid.uuid4().hex
    print(f"models={model_names} sweep={sweep.name} variant={variant.name} -> {out_dir}", flush=True)

    X_train, X_test, y_train, y_test, subject_test, provenance = load_har()
    models, fit_seconds, performance, fingerprints = {}, {}, {}, {}
    for name in model_names:
        model = config.MODEL_BUILDERS[name]()
        started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds[name] = perf_counter() - started
        performance[name] = evaluate_predictive_performance(model, X_test, y_test)
        fingerprints[name] = model_fingerprint(model, X_test)
        models[name] = model
        print(f"[{name}] fitted in {fit_seconds[name]:.1f}s; test metrics {performance[name]}", flush=True)

    benchmark_started = perf_counter()
    tables, probes = benchmark_lime(
        models, X_train, y_train, X_test, y_test, subject_test, sweep, variant
    )
    benchmark_seconds = perf_counter() - benchmark_started

    shared_meta = {
        "run_id": run_id,
        "models_in_run": model_names,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "version_check": version_check,
        "thread_env": thread_env_snapshot(),
        "config": config.snapshot(sweep, variant),
        "data": provenance,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "benchmark_seconds": benchmark_seconds,
        "speed_probes": probes,
    }
    for name in model_names:
        model_tables = tables[name]
        summary = summarise(model_tables["raw"], model_tables["stability"])
        validate(model_tables["raw"], model_tables["stability"], model_tables["references"], summary, sweep)
        run_meta = {
            **shared_meta,
            "model": name,
            "model_params": models[name].get_params(deep=True),
            "model_fingerprint": fingerprints[name],
            "explained_cases": sorted(int(i) for i in model_tables["raw"]["case_index"].unique()),
            "fit_seconds": fit_seconds[name],
        }
        save_results(out_dir, name, {**model_tables, "summary": summary}, performance[name], run_meta)
    print(f"done in {benchmark_seconds:.1f}s; results in {out_dir}", flush=True)


def cli(default_models: list[str]) -> None:
    """Entry point shared by run_all (all models) and the per-model debug scripts."""
    parser = argparse.ArgumentParser(description="Controlled LIME benchmark")
    parser.add_argument("--models", nargs="+", choices=config.MODEL_NAMES, default=default_models)
    parser.add_argument(
        "--quick", action="store_true", help="tiny smoke-test sweep written to results/quick"
    )
    parser.add_argument(
        "--variant", choices=list(config.VARIANTS), default=config.MAIN_VARIANT.name
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="run a full sweep outside the pinned environment (not for the paper)",
    )
    args = parser.parse_args()
    run_experiment(
        args.models,
        quick=args.quick,
        variant_name=args.variant,
        allow_version_mismatch=args.allow_version_mismatch,
    )


if __name__ == "__main__":
    cli(list(config.MODEL_NAMES))
