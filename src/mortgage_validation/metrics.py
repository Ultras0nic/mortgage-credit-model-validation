"""Transparent metric implementations used by independent validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, probability)
    return float(np.max(tpr - fpr))


def calibration_intercept_slope(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Fit y ~ intercept + slope*logit(p) using two-parameter Newton IRLS."""
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    y = np.asarray(y_true, dtype=float)
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        eta = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        weights = np.clip(mu * (1 - mu), 1e-8, None)
        hessian = x.T @ (weights[:, None] * x)
        score = x.T @ (y - mu)
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def bootstrap_interval(
    y_true: np.ndarray, probability: np.ndarray, metric, n_boot: int = 500, seed: int = 20260713
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    y = np.asarray(y_true)
    p = np.asarray(probability)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(float(metric(y[idx], p[idx])))
    if not values:
        raise ValueError("Bootstrap requires both outcome classes in at least one resample")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_difference(
    y_true: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    metric,
    n_boot: int = 500,
    seed: int = 20260713,
) -> tuple[float, float]:
    """Percentile interval for second-minus-first performance on paired loans."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    first = np.asarray(first_probability)
    second = np.asarray(second_probability)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(float(metric(y[idx], second[idx]) - metric(y[idx], first[idx])))
    if not values:
        raise ValueError("Bootstrap requires both outcome classes in at least one resample")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def wilson_interval(defaults: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for an observed binomial event rate."""
    if n <= 0 or not 0 <= defaults <= n:
        raise ValueError("defaults and n must satisfy 0 <= defaults <= n with n > 0")
    rate = defaults / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def metric_row(y_true: np.ndarray, probability: np.ndarray, split: str) -> dict[str, float | str]:
    y = np.asarray(y_true)
    p = np.asarray(probability)
    auc = float(roc_auc_score(y, p))
    intercept, slope = calibration_intercept_slope(y, p)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_null = prevalence * (1 - prevalence)
    return {
        "split": split,
        "n": int(len(y)),
        "defaults": int(y.sum()),
        "observed_rate": prevalence,
        "predicted_rate": float(p.mean()),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "brier": brier,
        "brier_null": brier_null,
        "brier_skill": 1 - brier / brier_null if brier_null > 0 else np.nan,
        "log_loss": float(log_loss(y, p)),
        "oe_ratio": float(y.sum() / p.sum()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def calibration_table(y_true: pd.Series, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": probability})
    frame["risk_decile"] = pd.qcut(frame["predicted"], bins, labels=False, duplicates="drop") + 1
    table = (
        frame.groupby("risk_decile", observed=True)
        .agg(n=("actual", "size"), defaults=("actual", "sum"), observed_rate=("actual", "mean"), predicted_rate=("predicted", "mean"))
        .reset_index()
    )
    intervals = [wilson_interval(int(row.defaults), int(row.n)) for row in table.itertuples()]
    table["observed_rate_ci_low"] = [interval[0] for interval in intervals]
    table["observed_rate_ci_high"] = [interval[1] for interval in intervals]
    return table


def population_stability_index(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce")
    comp = pd.to_numeric(comparison, errors="coerce")
    valid_reference = ref.dropna()
    if valid_reference.empty:
        raise ValueError("PSI reference sample must contain at least one non-missing value")
    edges = np.unique(np.quantile(valid_reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False).to_numpy()
    comp_counts = pd.cut(comp, edges, include_lowest=True).value_counts(sort=False).to_numpy()
    ref_counts = np.append(ref_counts, ref.isna().sum())
    comp_counts = np.append(comp_counts, comp.isna().sum())
    eps = 1e-6
    ref_pct = np.clip(ref_counts / len(ref), eps, None)
    comp_pct = np.clip(comp_counts / len(comp), eps, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))
