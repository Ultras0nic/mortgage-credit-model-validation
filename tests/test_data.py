from pathlib import Path

from mortgage_validation.data import generate_synthetic_mortgages, run_sql_quality_checks


ROOT = Path(__file__).resolve().parents[1]


def test_generation_is_deterministic():
    first = generate_synthetic_mortgages(n_loans=500, seed=11)
    second = generate_synthetic_mortgages(n_loans=500, seed=11)
    assert first.equals(second)


def test_schema_ranges_and_chronological_splits():
    df = generate_synthetic_mortgages(n_loans=2_000, seed=12)
    assert df["loan_id"].is_unique
    assert set(df["default_12m"].unique()) <= {0, 1}
    assert df["fico_score"].between(300, 850).all()
    assert df["ltv"].between(20, 150).all()
    years = df["origination_date"].dt.year
    assert (df.loc[df["split"] == "train", "origination_date"].dt.year <= 2020).all()
    assert (years[df["split"] == "validation"] == 2021).all()
    assert (years[df["split"] == "oot"] == 2022).all()
    assert (df["performance_window_end"] >= df["origination_date"] + __import__("pandas").DateOffset(months=12)).all()


def test_sql_checks_pass_and_detect_duplicate():
    df = generate_synthetic_mortgages(n_loans=1_000, seed=13)
    sql_path = ROOT / "sql" / "data_quality_checks.sql"
    passed = run_sql_quality_checks(df, sql_path)
    assert (passed["status"] == "PASS").all()
    broken = __import__("pandas").concat([df, df.iloc[[0]]], ignore_index=True)
    failed = run_sql_quality_checks(broken, sql_path)
    duplicate = failed.loc[failed["check"] == "duplicate_loan_id"].iloc[0]
    assert duplicate["status"] == "FAIL"
    assert duplicate["failed_rows"] == 1


def test_sql_checks_detect_missing_input_and_added_domains():
    df = generate_synthetic_mortgages(n_loans=1_000, seed=14)
    sql_path = ROOT / "sql" / "data_quality_checks.sql"
    df.loc[df.index[0], "fico_score"] = None
    df.loc[df.index[1], "rate_spread"] = 11
    df.loc[df.index[2], "prior_delinquency"] = 2
    failed = run_sql_quality_checks(df, sql_path).set_index("check")
    assert failed.loc["missing_model_input", "status"] == "FAIL"
    assert failed.loc["invalid_ranges", "status"] == "FAIL"


def test_sql_checks_enforce_data_as_of_date():
    df = generate_synthetic_mortgages(n_loans=1_000, seed=15)
    sql_path = ROOT / "sql" / "data_quality_checks.sql"
    df.loc[df.index[0], "performance_window_end"] = __import__("pandas").Timestamp("2024-01-01")
    failed = run_sql_quality_checks(df, sql_path).set_index("check")
    assert failed.loc["incomplete_outcome_window", "status"] == "FAIL"
