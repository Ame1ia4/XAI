"""Case-level aggregation, bootstrap CIs, paired tests with Holm correction, and runtime scaling.

The explained case is the independent unit. Repeats (and pairs of repeats) of one
case share that case, so they are averaged within the case first; uncertainty then
comes from a class-stratified bootstrap over cases. Every model explains the same
cases (and the same class of each case), so model differences are paired by case:
they use the same bootstrap resamples and a sign-flip permutation test.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd

from lime_study import config

GROUP_COLUMNS = ["model", "feature_count", "num_samples"]
CASE_COLUMNS = GROUP_COLUMNS + ["case_index"]
JACCARD = f"top{config.TOP_K}_jaccard"

# Metric name -> label used in tables and figures.
METRICS = {
    "runtime_seconds": "Runtime per explanation (s)",
    "fidelity_heldout": "Held-out surrogate fit (weighted R², flat cases excluded)",
    "heldout_rmse": "Held-out surrogate error (weighted RMSE, probability units)",
    "fidelity_insample": "In-sample surrogate fit (LIME score)",
    "reference_jaccard": f"Top-{config.TOP_K} Jaccard vs large-N reference",
    "reference_ceiling": f"Top-{config.TOP_K} Jaccard between the two references",
    "repeat_jaccard": f"Top-{config.TOP_K} Jaccard between repeats",
}
# Constant across budgets (one value per case x feature count), so not compared per budget.
PER_FEATURE_COUNT_METRICS = ("reference_ceiling",)
PAIRED_METRICS = [m for m in METRICS if m not in PER_FEATURE_COUNT_METRICS]


# --- Case level -----------------------------------------------------------
def case_level(raw: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """One row per model x feature count x budget x case, every metric averaged in-case."""
    per_run = raw.groupby(CASE_COLUMNS, as_index=False).agg(
        true_class=("true_class", "first"),
        correct=("correct", "first"),
        runtime_seconds=("runtime_seconds", "mean"),
        fidelity_heldout=("fidelity_heldout", "mean"),
        heldout_rmse=("heldout_rmse", "mean"),
        fidelity_insample=("fidelity_insample", "mean"),
        reference_jaccard=("reference_jaccard", "mean"),
        reference_ceiling=("reference_ceiling", "first"),
        target_sd=("target_sd", "first"),
        flat_output=("flat_output", "first"),
    )
    # R² divides by the model's local output variation; undefined in spirit when it is ~0.
    per_run.loc[per_run["flat_output"].astype(bool), "fidelity_heldout"] = np.nan
    per_pair = stability.groupby(CASE_COLUMNS, as_index=False).agg(
        repeat_jaccard=(JACCARD, "mean")
    )
    cases = per_run.merge(per_pair, on=CASE_COLUMNS, validate="one_to_one")
    if len(cases) != len(per_run):
        raise AssertionError("Stability rows do not cover every case")
    cases["underdetermined"] = [
        config.underdetermined(n, d) for n, d in zip(cases["num_samples"], cases["feature_count"])
    ]
    return cases


# --- Resampling and tests -------------------------------------------------
def _case_classes(cases: pd.DataFrame) -> pd.Series:
    return cases.groupby("case_index")["true_class"].first().sort_index()


def bootstrap_indices(case_classes: pd.Series) -> np.ndarray:
    """(resamples x cases) positions, resampling cases within each true class.

    The same matrix is reused for every group, model and fit, so all comparisons are paired.
    """
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    classes = case_classes.to_numpy()
    strata = [np.flatnonzero(classes == label) for label in np.unique(classes)]
    return np.column_stack(
        [
            positions[rng.integers(0, len(positions), size=(config.BOOTSTRAP_RESAMPLES, len(positions)))]
            for positions in strata
        ]
    )


def _percentiles(boot: np.ndarray) -> tuple[float, float]:
    tail = (1 - config.CI_LEVEL) / 2 * 100
    low, high = np.nanpercentile(boot, [tail, 100 - tail], axis=0)
    return float(low), float(high)


def _ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    with np.errstate(all="ignore"):
        return _percentiles(np.nanmean(values[indices], axis=1))


def sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided paired permutation test of mean difference = 0 (randomly flipping signs).

    Exact when 2**n is small enough to enumerate; otherwise Monte Carlo with the
    (b + 1) / (B + 1) estimate, which is never zero.
    """
    diff = differences[~np.isnan(differences)]
    n = len(diff)
    if n == 0:
        return np.nan
    observed = abs(diff.mean())
    tolerance = 1e-12 * max(1.0, observed)
    if 2**n <= config.PERMUTATION_RESAMPLES:
        signs = np.array(list(product((1.0, -1.0), repeat=n)))
        return float(np.mean(np.abs(signs @ diff / n) >= observed - tolerance))
    rng = np.random.default_rng(config.PERMUTATION_SEED)
    signs = rng.choice((1.0, -1.0), size=(config.PERMUTATION_RESAMPLES, n))
    extreme = int(np.sum(np.abs(signs @ diff / n) >= observed - tolerance))
    return (extreme + 1) / (config.PERMUTATION_RESAMPLES + 1)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (family = the non-missing entries)."""
    p = p_values.to_numpy(dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    running = 0.0
    for rank, position in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, min(1.0, (len(valid) - rank) * p[position]))
        adjusted[position] = running
    return pd.Series(adjusted, index=p_values.index)


def passes_reference_rule(mean_ceiling: float) -> bool:
    """Mean reference self-agreement >= the pre-set rule; the tolerance stops float
    rounding (e.g. a mean of exactly 4/6) from flipping the decision."""
    return bool(mean_ceiling >= config.REFERENCE_MIN_AGREEMENT - 1e-9)


def reference_validity(cases: pd.DataFrame) -> dict[tuple[str, int], bool]:
    """(model, feature_count) -> do the two references agree enough to act as a reference?

    The ceiling is constant across budgets, so one budget's rows hold every case's value.
    """
    one_budget = cases[cases["num_samples"] == cases["num_samples"].min()]
    means = one_budget.groupby(["model", "feature_count"])["reference_ceiling"].mean()
    return {key: passes_reference_rule(value) for key, value in means.items()}


def primary_budget(budgets) -> int:
    budgets = sorted(set(budgets))
    return config.PRIMARY_BUDGET if config.PRIMARY_BUDGET in budgets else budgets[-1]


# --- Summaries ------------------------------------------------------------
def summarise(cases: pd.DataFrame) -> pd.DataFrame:
    """Mean and bootstrap CI of every metric per model x feature count x budget."""
    case_classes = _case_classes(cases)
    indices = bootstrap_indices(case_classes)
    rows = []
    for keys, group in cases.groupby(GROUP_COLUMNS, sort=True):
        group = group.set_index("case_index").reindex(case_classes.index)
        row = dict(zip(GROUP_COLUMNS, keys))
        row["cases"] = int(group["runtime_seconds"].notna().sum())
        row["underdetermined"] = config.underdetermined(row["num_samples"], row["feature_count"])
        row["correct_case_fraction"] = float(group["correct"].astype(float).mean())
        row["flat_cases"] = int(group["flat_output"].astype(bool).sum())
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _ci(values, indices)
        # Agreement relative to how well two references agree with each other (ratio of means).
        reference = group["reference_jaccard"].to_numpy(dtype=float)
        ceiling = group["reference_ceiling"].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            row["reference_jaccard_relative_mean"] = float(np.nanmean(reference) / np.nanmean(ceiling))
            boot = np.nanmean(reference[indices], axis=1) / np.nanmean(ceiling[indices], axis=1)
        (
            row["reference_jaccard_relative_ci_low"],
            row["reference_jaccard_relative_ci_high"],
        ) = _percentiles(boot)
        row["reference_valid"] = passes_reference_rule(row["reference_ceiling_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(cases: pd.DataFrame) -> pd.DataFrame:
    """Model A minus model B, per case, for every model pair, setting and metric.

    p-values: sign-flip permutation test. `p_holm_primary` is Holm within the
    pre-specified primary family; `p_holm_all` is Holm over the whole table.
    Reference-Jaccard comparisons leave the primary family where either model's
    references fail the pre-specified agreement rule (config.REFERENCE_MIN_AGREEMENT).
    """
    case_classes = _case_classes(cases)
    indices = bootstrap_indices(case_classes)
    models = [m for m in config.MODEL_NAMES if m in set(cases["model"])]
    budget = primary_budget(cases["num_samples"])
    valid = reference_validity(cases)
    wide = cases.pivot_table(
        index=["feature_count", "num_samples", "case_index"],
        columns="model",
        values=PAIRED_METRICS,
        dropna=False,
    )
    rows = []
    for (feature_count, num_samples), block in wide.groupby(level=[0, 1], sort=True):
        block = block.droplevel([0, 1]).reindex(case_classes.index)
        for first, second in combinations(models, 2):
            for metric in PAIRED_METRICS:
                diff = (block[(metric, first)] - block[(metric, second)]).to_numpy(dtype=float)
                low, high = _ci(diff, indices)
                references_ok = (
                    valid[(first, feature_count)] and valid[(second, feature_count)]
                    if metric == "reference_jaccard"
                    else None
                )
                rows.append(
                    {
                        "feature_count": feature_count,
                        "num_samples": num_samples,
                        "underdetermined": config.underdetermined(num_samples, feature_count),
                        "metric": metric,
                        "model_a": first,
                        "model_b": second,
                        "cases_compared": int(np.isfinite(diff).sum()),
                        "mean_difference": float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": sign_flip_p(diff),
                        "reference_valid": references_ok,
                        "primary": bool(
                            num_samples == budget
                            and metric in config.PRIMARY_METRICS
                            and not config.underdetermined(num_samples, feature_count)
                            and references_ok is not False
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    table["p_holm_all"] = holm(table["p_value"])
    table["p_holm_primary"] = holm(table["p_value"].where(table["primary"]))
    table["significant_primary"] = table["p_holm_primary"] < config.ALPHA
    return table


def reference_check(cases: pd.DataFrame) -> pd.DataFrame:
    """Is the reference itself less stable at larger d?

    For each model and pair of feature counts, compares the change in reference
    self-agreement (the ceiling) with the change in LIME's reference and repeat
    agreement at the primary budget. If the ceiling drops as much as LIME's agreement,
    the drop is partly reference noise, not LIME getting worse. Holm over the table.
    """
    case_classes = _case_classes(cases)
    indices = bootstrap_indices(case_classes)
    budget = primary_budget(cases["num_samples"])
    at_budget = cases[cases["num_samples"] == budget]
    metrics = ("reference_ceiling", "reference_jaccard", "repeat_jaccard")
    rows = []
    for model in [m for m in config.MODEL_NAMES if m in set(cases["model"])]:
        wide = (
            at_budget[at_budget["model"] == model]
            .pivot(index="case_index", columns="feature_count", values=list(metrics))
            .reindex(case_classes.index)
        )
        for metric in metrics:
            for d in config.FEATURE_COUNTS:
                values = wide[(metric, d)].to_numpy(dtype=float)
                low, high = _ci(values, indices)
                rows.append(
                    {
                        "model": model,
                        "num_samples": budget,
                        "metric": metric,
                        "comparison": f"d={d}",
                        "estimate": float(np.nanmean(values)),
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": np.nan,
                        "reference_valid": (
                            passes_reference_rule(np.nanmean(values))
                            if metric == "reference_ceiling"
                            else None
                        ),
                    }
                )
            for small, large in combinations(config.FEATURE_COUNTS, 2):
                diff = (wide[(metric, large)] - wide[(metric, small)]).to_numpy(dtype=float)
                low, high = _ci(diff, indices)
                rows.append(
                    {
                        "model": model,
                        "num_samples": budget,
                        "metric": metric,
                        "comparison": f"d={large} minus d={small}",
                        "estimate": float(np.nanmean(diff)),
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": sign_flip_p(diff),
                    }
                )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_value"])
    return table


# --- Runtime scaling (research question 1) --------------------------------
def runtime_scaling(cases: pd.DataFrame) -> pd.DataFrame:
    """Power-law fits of runtime: log t = a + b·log N + c·log d (b, c are scaling exponents).

    Fitted by least squares on per-case runtimes (repeats averaged). Every case has the
    same design points, so the fit equals OLS on the across-case mean of log runtime,
    which makes a class-stratified case bootstrap cheap. Also reported: the N exponent
    within each d, the d exponent within each N, an interaction model (does the N
    exponent change with d?), and paired exponent differences between models.
    """
    case_classes = _case_classes(cases)
    indices = bootstrap_indices(case_classes)
    budgets = sorted(cases["num_samples"].unique())
    counts = list(config.FEATURE_COUNTS)
    log_n, log_d = np.log(budgets), np.log(counts)
    grid_d, grid_n = np.meshgrid(log_d, log_n, indexing="ij")  # shape (d, N)
    flat_d, flat_n = grid_d.ravel(), grid_n.ravel()
    designs = {
        "additive": np.column_stack([np.ones_like(flat_n), flat_n, flat_d]),
        "interaction": np.column_stack([np.ones_like(flat_n), flat_n, flat_d, flat_n * flat_d]),
    }
    terms = {"additive": ["intercept", "log_N", "log_d"], "interaction": ["intercept", "log_N", "log_d", "log_N:log_d"]}

    def log_runtime(model: str) -> np.ndarray:
        """(cases, d, N) array of log mean runtime."""
        table = cases[cases["model"] == model].pivot_table(
            index="case_index", columns=["feature_count", "num_samples"], values="runtime_seconds"
        )
        table = table.reindex(case_classes.index)
        return np.log(table[[(d, n) for d in counts for n in budgets]].to_numpy()).reshape(
            len(case_classes), len(counts), len(budgets)
        )

    def fits(y: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
        """y: (..., d, N) mean log runtime -> estimates keyed by (fit, term); vectorised over leading dims."""
        flat = y.reshape(*y.shape[:-2], -1)
        out = {}
        for fit, design in designs.items():
            beta = flat @ np.linalg.pinv(design).T
            for i, term in enumerate(terms[fit]):
                out[(fit, term)] = beta[..., i]
            residual = flat - beta @ design.T
            total = flat - flat.mean(axis=-1, keepdims=True)
            out[(fit, "r2")] = 1 - (residual**2).sum(-1) / (total**2).sum(-1)
        for j, d in enumerate(counts):
            design = np.column_stack([np.ones_like(log_n), log_n])
            out[(f"within d={d}", "log_N")] = (y[..., j, :] @ np.linalg.pinv(design).T)[..., 1]
        for k, n in enumerate(budgets):
            design = np.column_stack([np.ones_like(log_d), log_d])
            out[(f"within N={n}", "log_d")] = (y[..., :, k] @ np.linalg.pinv(design).T)[..., 1]
        return out

    models = [m for m in config.MODEL_NAMES if m in set(cases["model"])]
    point, boot = {}, {}
    for model in models:
        y = log_runtime(model)
        point[model] = fits(y.mean(axis=0))
        boot[model] = fits(y[indices].mean(axis=1))

    rows = []

    def add(label: str, estimates: dict, samples: dict) -> None:
        for (fit, term), value in estimates.items():
            low, high = _percentiles(samples[(fit, term)])
            is_exponent = term in ("log_N", "log_d")
            rows.append(
                {
                    "model": label,
                    "fit": fit,
                    "term": term,
                    "estimate": float(value),
                    "ci_low": low,
                    "ci_high": high,
                    # Runtime multiplier when N (or d) doubles: 2**exponent.
                    "multiplier_per_doubling": float(2**value) if is_exponent else np.nan,
                }
            )

    for model in models:
        add(model, point[model], boot[model])
    for first, second in combinations(models, 2):
        keys = [k for k in point[first] if k[1] in ("log_N", "log_d", "log_N:log_d", "intercept")]
        add(
            f"{first} - {second}",
            {k: point[first][k] - point[second][k] for k in keys},
            {k: boot[first][k] - boot[second][k] for k in keys},
        )
    table = pd.DataFrame(rows)
    differences = table["model"].str.contains(" - ")
    table.loc[differences, "multiplier_per_doubling"] = np.nan
    return table
