from pathlib import Path

from mortgage_validation.data import generate_synthetic_mortgages, run_sql_quality_checks
from mortgage_validation.modeling import FEATURES, champion_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_model_pipeline_scores_without_leakage_fields():
    df = generate_synthetic_mortgages(n_loans=3_000, seed=21)
    train = df[df["split"] == "train"]
    oot = df[df["split"] == "oot"]
    model = champion_pipeline().fit(train[FEATURES], train["default_12m"])
    scores = model.predict_proba(oot[FEATURES])[:, 1]
    assert len(scores) == len(oot)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert "default_12m" not in FEATURES
    assert "performance_window_end" not in FEATURES
    assert (run_sql_quality_checks(df, ROOT / "sql" / "data_quality_checks.sql")["status"] == "PASS").all()

