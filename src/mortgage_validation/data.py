"""Deterministic synthetic mortgage data and embedded SQL quality checks."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260713
N_LOANS = 20_000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_mortgages(n_loans: int = N_LOANS, seed: int = SEED) -> pd.DataFrame:
    """Generate mortgage-like loans with a documented nonlinear default DGP.

    The 2022 cohort contains a modest adverse mix/economic shift and intercept
    shift, allowing an out-of-time validation to detect calibration and drift.
    """
    rng = np.random.default_rng(seed)
    start = np.datetime64("2018-01-01")
    end = np.datetime64("2023-01-01")
    days = (end - start).astype(int)
    dates = start + rng.integers(0, days, n_loans).astype("timedelta64[D]")
    dates = pd.to_datetime(dates)
    year = dates.year.to_numpy()
    oot = year == 2022
    validation = year == 2021
    quality = rng.normal(0, 1, n_loans)

    fico = np.clip(715 + 43 * quality - 10 * oot + rng.normal(0, 25, n_loans), 520, 850)
    ltv = np.clip(78 - 7 * quality + 4 * oot + rng.normal(0, 11, n_loans), 25, 145)
    dti = np.clip(35 - 3 * quality + 2 * oot + rng.normal(0, 8, n_loans), 5, 75)
    unemployment = np.clip(
        4.4 + 0.10 * (year - 2018) + 0.35 * oot + rng.normal(0, 0.55, n_loans), 2, 15
    )
    hpi = np.clip(6.5 - 0.50 * (year - 2018) - 2.0 * oot + rng.normal(0, 2.0, n_loans), -20, 25)
    rate_spread = np.clip(1.2 - 0.22 * quality + 0.2 * oot + rng.normal(0, 0.45, n_loans), -1, 10)
    interest_rate = np.clip(3.3 + 0.12 * (year - 2018) + rate_spread + 0.30 * oot, 1, 15)
    amount = np.clip(rng.lognormal(np.log(235_000), 0.48, n_loans), 25_000, 1_500_000)

    purpose = rng.choice(
        ["purchase", "rate_term_refi", "cash_out_refi"], n_loans, p=[0.58, 0.25, 0.17]
    )
    occupancy = rng.choice(["owner", "second_home", "investor"], n_loans, p=[0.82, 0.07, 0.11])
    state = rng.choice(["IL", "WI", "OTHER"], n_loans, p=[0.53, 0.32, 0.15])
    prior_delinq = rng.binomial(1, _sigmoid(-2.15 - 0.65 * quality + 0.2 * oot))

    # Logistic DGP; champion intentionally omits the interaction term.
    log_odds = (
        -3.55
        - 0.012 * (fico - 700)
        + 0.026 * (ltv - 80)
        + 0.020 * (dti - 35)
        + 0.22 * rate_spread
        + 0.20 * (unemployment - 5)
        - 0.045 * hpi
        + 1.05 * prior_delinq
        + 0.35 * (purpose == "cash_out_refi")
        + 0.38 * (occupancy == "investor")
        + 0.55 * ((ltv > 95) & (fico < 680))
        + 0.18 * oot
    )
    default = rng.binomial(1, _sigmoid(log_odds))
    split = np.where(oot, "oot", np.where(validation, "validation", "train"))

    df = pd.DataFrame(
        {
            "loan_id": [f"SYN{i:07d}" for i in range(1, n_loans + 1)],
            "origination_date": dates,
            "performance_window_end": dates + pd.DateOffset(months=12),
            "split": split,
            "fico_score": np.round(fico, 0),
            "ltv": np.round(ltv, 2),
            "dti": np.round(dti, 2),
            "loan_amount": np.round(amount, 2),
            "interest_rate": np.round(interest_rate, 3),
            "rate_spread": np.round(rate_spread, 3),
            "loan_purpose": purpose,
            "occupancy_status": occupancy,
            "prior_delinquency": prior_delinq,
            "state": state,
            "unemployment_rate": np.round(unemployment, 2),
            "hpi_yoy_change": np.round(hpi, 2),
            "default_12m": default,
        }
    )
    return df.sort_values(["origination_date", "loan_id"]).reset_index(drop=True)


def run_sql_quality_checks(df: pd.DataFrame, sql_path: Path) -> pd.DataFrame:
    """Execute named zero-row SQLite checks and return auditable results."""
    text = sql_path.read_text(encoding="utf-8")
    blocks = re.split(r"-- check:\s*([a-z0-9_]+)\s*\n", text, flags=re.IGNORECASE)
    results: list[dict[str, object]] = []
    sql_df = df.copy()
    for col in ["origination_date", "performance_window_end"]:
        sql_df[col] = pd.to_datetime(sql_df[col]).dt.strftime("%Y-%m-%d")
    with sqlite3.connect(":memory:") as connection:
        sql_df.to_sql("loans", connection, index=False)
        for i in range(1, len(blocks), 2):
            name, query = blocks[i], blocks[i + 1].strip()
            failures = pd.read_sql_query(query, connection)
            results.append(
                {"check": name, "failed_rows": len(failures), "status": "PASS" if failures.empty else "FAIL"}
            )
    return pd.DataFrame(results)


def write_generated_data(project_root: Path, n_loans: int = N_LOANS, seed: int = SEED) -> pd.DataFrame:
    df = generate_synthetic_mortgages(n_loans=n_loans, seed=seed)
    output_dir = project_root / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "mortgage_loans.csv", index=False, date_format="%Y-%m-%d")
    return df
