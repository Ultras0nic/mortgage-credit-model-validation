"""Small set of decision-useful validation charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

COLOR = "#24527a"
ACCENT = "#c4512d"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def discrimination_chart(y: np.ndarray, champion: np.ndarray, challenger: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, pred, color in [("Champion logistic", champion, COLOR), ("Challenger tree", challenger, ACCENT)]:
        fpr, tpr, _ = roc_curve(y, pred)
        axes[0].plot(fpr, tpr, label=label, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="OOT ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend()
    order = np.argsort(champion)
    sorted_y = y[order]
    nondefault = 1 - sorted_y
    cum_default = np.cumsum(sorted_y) / sorted_y.sum()
    cum_nondefault = np.cumsum(nondefault) / nondefault.sum()
    axes[1].plot(np.linspace(0, 1, len(y)), cum_default, label="Defaults", color=ACCENT)
    axes[1].plot(np.linspace(0, 1, len(y)), cum_nondefault, label="Non-defaults", color=COLOR)
    axes[1].set(title="OOT KS separation", xlabel="Share of loans, low to high PD", ylabel="Cumulative share")
    axes[1].legend()
    _save(fig, path)


def calibration_chart(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(table["predicted_rate"], table["observed_rate"], "o-", color=COLOR)
    if {"observed_rate_ci_low", "observed_rate_ci_high"} <= set(table.columns):
        errors = np.vstack(
            [
                table["observed_rate"] - table["observed_rate_ci_low"],
                table["observed_rate_ci_high"] - table["observed_rate"],
            ]
        )
        ax.errorbar(table["predicted_rate"], table["observed_rate"], yerr=errors, fmt="none", ecolor=COLOR, alpha=0.5, capsize=2)
    limit = max(table[["predicted_rate", "observed_rate"]].max()) * 1.08
    ax.plot([0, limit], [0, limit], "--", color="grey", label="Perfect calibration")
    ax.set(title="OOT calibration by PD decile", xlabel="Mean predicted default rate", ylabel="Observed default rate")
    ax.legend()
    _save(fig, path)


def backtest_chart(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    ax.plot(x, table["observed_rate"], "o-", color=ACCENT, label="Observed")
    if {"observed_rate_ci_low", "observed_rate_ci_high"} <= set(table.columns):
        errors = np.vstack(
            [
                table["observed_rate"] - table["observed_rate_ci_low"],
                table["observed_rate_ci_high"] - table["observed_rate"],
            ]
        )
        ax.errorbar(x, table["observed_rate"], yerr=errors, fmt="none", ecolor=ACCENT, alpha=0.5, capsize=3)
    ax.plot(x, table["predicted_rate"], "o-", color=COLOR, label="Predicted")
    ax.set_xticks(x, table["quarter"], rotation=45, ha="right")
    ax.set(title="OOT quarterly backtest", ylabel="Default rate")
    ax.legend()
    _save(fig, path)


def bar_chart(table: pd.DataFrame, label: str, value: str, title: str, path: Path, threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = table.sort_values(value)
    ax.barh(ordered[label], ordered[value], color=COLOR)
    if threshold is not None:
        ax.axvline(threshold, color=ACCENT, linestyle="--", label=f"Material threshold ({threshold:.2f})")
        ax.legend()
    ax.set(title=title, xlabel=value.replace("_", " ").title())
    _save(fig, path)


def comparison_chart(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].bar(table["model"], table["auc"], color=[COLOR, ACCENT])
    axes[0].set(title="OOT AUC", ylim=(0.5, 1.0))
    axes[1].bar(table["model"], table["brier"], color=[COLOR, ACCENT])
    axes[1].set(title="OOT Brier score (lower is better)")
    _save(fig, path)
