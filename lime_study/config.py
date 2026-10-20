"""Single source of truth for the controlled LIME cost/quality study.

Every value shared between models lives here. Model scripts only name a model;
they must not redefine data, LIME, or sweep settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
FEATURE_COUNTS = (100, 300, 500)  # nested LIME perturbation subsets; 500 = TabFM max_num_features


# --- Sweep ----------------------------------------------------------------
@dataclass(frozen=True)
class Sweep:
    name: str
    explanations_per_class: int
    sample_budgets: tuple[int, ...]
    repeats: int


FULL_SWEEP = Sweep("full", 3, (500, 1_000, 2_500, 5_000, 10_000), 5)
QUICK_SWEEP = Sweep("quick", 1, (500, 1_000), 2)  # smoke test only, never for the paper

NUM_FEATURES = 10  # features reported per explanation
TOP_K = 5  # top-k cutoff for Jaccard stability

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


def results_dir(sweep: Sweep) -> Path:
    return RESULTS_DIR if sweep is FULL_SWEEP else RESULTS_DIR / sweep.name


def figures_dir(sweep: Sweep) -> Path:
    return FIGURES_DIR if sweep is FULL_SWEEP else FIGURES_DIR / sweep.name


def snapshot(sweep: Sweep) -> dict[str, Any]:
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
        "num_features": NUM_FEATURES,
        "top_k": TOP_K,
        "lime_explainer_kwargs": LIME_EXPLAINER_KWARGS,
        "lime_distance_metric": LIME_DISTANCE_METRIC,
        "ridge_alpha": RIDGE_ALPHA,
        "n_jobs": N_JOBS,
    }
