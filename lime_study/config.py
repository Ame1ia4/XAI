"""Single source of truth for the controlled LIME cost/quality study.

Every value shared between models lives here. Model scripts only name a model;
they must not redefine data, LIME, or sweep settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
FIGURES_DIR = PACKAGE_DIR / "figures"

# --- Seeds ----------------------------------------------------------------
RANDOM_STATE = 42
BACKGROUND_SIZE = 2_000  # training rows passed to LimeTabularExplainer

# --- Dataset block: UCI Human Activity Recognition Using Smartphones -------
# 561 continuous features, 6 classes. Uses the official subject-based split
# (21 training volunteers / 7,352 rows, 9 test volunteers / 2,947 rows).
# Justification and sources: notes/dataset_justification.md
DATASET_NAME = "uci_har"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATA_DOI = "10.24432/C54S4K"
DATA_DIR = PACKAGE_DIR / "data"  # download cache (gitignored)
N_TRAIN_ROWS = 7_352
N_TEST_ROWS = 2_947
N_FEATURES = 561
CLASS_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
# Nested LIME perturbation subsets. 50 adds a fourth design point for the runtime
# exponent in d and never has a low sample-to-feature ratio; 500 = TabFM max_num_features.
FEATURE_COUNTS = (50, 100, 300, 500)


# --- Sweep ----------------------------------------------------------------
@dataclass(frozen=True)
class Sweep:
    name: str
    explanations_per_class: int
    sample_budgets: tuple[int, ...]
    repeats: int
    reference_samples: int  # num_samples of the large-N reference explanations


# 5 explained cases per true class (30 independent units); budgets unchanged around
# lime's default num_samples=5,000.
FULL_SWEEP = Sweep("full", 5, (500, 1_000, 2_500, 5_000, 10_000), 5, 50_000)
QUICK_SWEEP = Sweep("quick", 1, (500, 1_000), 2, 5_000)  # smoke test only, never for the paper

NUM_FEATURES = 10  # features reported per explanation
TOP_K = 5  # top-k cutoff for Jaccard stability and reference agreement

# --- Quality measures -----------------------------------------------------
# Held-out fidelity: the surrogate is scored on fresh perturbations it was not fitted on.
# One held-out set per case x feature count, shared by every budget, repeat and model.
HOLDOUT_SAMPLES = 5_000
HOLDOUT_SEED = RANDOM_STATE + 10_000
# Reference explanations: large-N LIME runs per case x feature count, with seeds disjoint
# from the repeat seeds. Agreement with them measures convergence, not just seed stability.
REFERENCE_SEEDS = (RANDOM_STATE + 20_000, RANDOM_STATE + 20_001)


def case_seed(base: int, case_index: int) -> int:
    """Seed for one explained case, derived from a base seed (repeat, held-out or reference).

    Cases are the independent units in every test, so each case gets its own random
    stream; the seed is shared by every model, feature count and budget of that case,
    which keeps those comparisons paired (common random numbers).
    """
    return int(np.random.SeedSequence([base, case_index]).generate_state(1)[0])

# Validation rule for the references, fixed before results: per model x feature count,
# the two references must share on average at least 4 of their top 5 features
# (top-5 Jaccard >= 4/6 = 0.667). Where they don't, agreement with the reference is not
# read as convergence and those comparisons leave the primary family.
REFERENCE_MIN_AGREEMENT = 4 / 6


def low_sample_ratio(num_samples: int, feature_count: int) -> bool:
    """N <= d: a low sample-to-feature ratio.

    LIME's feature selection fits a ridge regression over all d features from N samples,
    so with N <= d the choice of features is mostly noise. Not strictly underdetermined,
    because the ridge is regularised. These settings are reported separately and left
    out of trend lines.
    """
    return num_samples <= feature_count


# "Flat" output: the model's probability for the explained class has a kernel-weighted
# standard deviation below 1 percentage point over the held-out perturbations. R² is a
# ratio to that variation, so it is not reported for flat cases; RMSE still is.
# RMSE is absolute error in probability units: it is bounded by how much the model's
# output moves locally (target_sd), so compare it across models alongside target_sd.
FLAT_OUTPUT_SD = 0.01

# --- Statistics and run order ----------------------------------------------
# UCI HAR test rows come from 9 participants, and rows from one participant may be
# correlated. Case selection spreads cases across participants, at most this many each.
MAX_CASES_PER_SUBJECT = 4
# Uncertainty comes from a class-stratified bootstrap over explained cases, the unit of
# the tests; repeats and repeat pairs are averaged within each case first. A second,
# participant-clustered bootstrap gives the more conservative "_cluster" intervals.
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = RANDOM_STATE
CI_LEVEL = 0.95
# Paired model comparisons: sign-flip permutation test on per-case differences,
# Holm-corrected. Exact enumeration when 2**n_cases <= PERMUTATION_RESAMPLES.
PERMUTATION_RESAMPLES = 10_000
PERMUTATION_SEED = RANDOM_STATE
ALPHA = 0.05
# Primary family (fixed before seeing results): every model pair x feature count, at
# LIME's default num_samples, for one cost and two quality metrics. Everything else
# in paired_differences.csv is secondary/descriptive (Holm over the whole table).
PRIMARY_BUDGET = 5_000  # lime's default num_samples; quick sweep falls back to its largest budget
PRIMARY_METRICS = ("runtime_seconds", "heldout_rmse", "reference_jaccard")
# Timed explanations run as a randomized complete block design (all models in one
# process; blocks in seeded random order; models in random order within each block).
RUN_ORDER_SEED = RANDOM_STATE
SPEED_PROBES = 10  # drift probes spread through the timed phase (plus one at the start)
SPEED_PROBE_WARMUPS = 3  # unrecorded probe runs before the first one, so it is not a cold start

# --- LIME settings (explicit, matching lime==0.2.0.1 defaults) ------------
LIME_EXPLAINER_KWARGS: dict[str, Any] = {
    "mode": "classification",
    "kernel_width": None,  # LIME default: 0.75 * sqrt(number of perturbed features)
    "kernel": None,  # exponential kernel
    "discretize_continuous": True,
    "discretizer": "quartile",
    "feature_selection": "auto",  # resolves to highest_weights when NUM_FEATURES > 6
    "sample_around_instance": False,
    "verbose": False,
}
LIME_DISTANCE_METRIC = "euclidean"
RIDGE_ALPHA = 1.0


# --- Variants: the main study plus two sensitivity checks -----------------
@dataclass(frozen=True)
class Variant:
    name: str
    subset: str  # "random": seeded permutation; "ranked": by ANOVA F on the training set
    kernel_width: float | None  # None: LIME default 0.75 * sqrt(feature_count)
    description: str


# Fixed width = LIME's default at d=300, so d=300 is unchanged and every other d changes.
FIXED_KERNEL_WIDTH = 0.75 * sqrt(300)
MAIN_VARIANT = Variant("main", "random", None, "Random nested subsets, LIME default kernel width")
VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        MAIN_VARIANT,
        replace(
            MAIN_VARIANT,
            name="ranked_subset",
            subset="ranked",
            description="Model-agnostic informative subsets (ANOVA F on training labels)",
        ),
        replace(
            MAIN_VARIANT,
            name="fixed_kernel",
            kernel_width=FIXED_KERNEL_WIDTH,
            description=f"Kernel width fixed at {FIXED_KERNEL_WIDTH:.3f} for every feature count",
        ),
    )
}


def lime_explainer_kwargs(variant: Variant) -> dict[str, Any]:
    return {**LIME_EXPLAINER_KWARGS, "kernel_width": variant.kernel_width}


def make_surrogate(seed: int) -> Ridge:
    """LIME's default local surrogate, constructed explicitly."""
    return Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=seed)


# --- Models ---------------------------------------------------------------
N_JOBS = 1  # single-threaded inference for every model so runtimes are comparable


def build_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def build_logistic_regression() -> Pipeline:
    # lbfgs ignores n_jobs; single threading comes from the BLAS env vars in threads.py.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS
    )


MODEL_BUILDERS: dict[str, Callable[[], Any]] = {
    "XGBoost": build_xgboost,
    "LogisticRegression": build_logistic_regression,
    "RandomForest": build_random_forest,
}
MODEL_NAMES = tuple(MODEL_BUILDERS)


def _output_dir(root: Path, sweep: Sweep, variant: Variant) -> Path:
    path = root if sweep is FULL_SWEEP else root / sweep.name
    return path if variant is MAIN_VARIANT else path / variant.name


def results_dir(sweep: Sweep, variant: Variant = MAIN_VARIANT) -> Path:
    return _output_dir(RESULTS_DIR, sweep, variant)


def figures_dir(sweep: Sweep, variant: Variant = MAIN_VARIANT) -> Path:
    return _output_dir(FIGURES_DIR, sweep, variant)


def snapshot(sweep: Sweep, variant: Variant = MAIN_VARIANT) -> dict[str, Any]:
    """JSON-serialisable record of every shared setting, saved with each run."""
    return {
        "dataset": DATASET_NAME,
        "data_url": DATA_URL,
        "data_doi": DATA_DOI,
        "random_state": RANDOM_STATE,
        "background_size": BACKGROUND_SIZE,
        "class_names": list(CLASS_NAMES),
        "feature_counts": list(FEATURE_COUNTS),
        "sweep": asdict(sweep),
        "variant": asdict(variant),
        "num_features": NUM_FEATURES,
        "top_k": TOP_K,
        "holdout_samples": HOLDOUT_SAMPLES,
        "holdout_seed": HOLDOUT_SEED,
        "reference_seeds": list(REFERENCE_SEEDS),
        "reference_min_agreement": REFERENCE_MIN_AGREEMENT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_level": CI_LEVEL,
        "run_order_seed": RUN_ORDER_SEED,
        "speed_probes": SPEED_PROBES,
        "speed_probe_warmups": SPEED_PROBE_WARMUPS,
        "max_cases_per_subject": MAX_CASES_PER_SUBJECT,
        "flat_output_sd": FLAT_OUTPUT_SD,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "permutation_seed": PERMUTATION_SEED,
        "alpha": ALPHA,
        "primary_budget": PRIMARY_BUDGET,
        "primary_metrics": list(PRIMARY_METRICS),
        "lime_explainer_kwargs": lime_explainer_kwargs(variant),
        "lime_distance_metric": LIME_DISTANCE_METRIC,
        "ridge_alpha": RIDGE_ALPHA,
        "n_jobs": N_JOBS,
    }
