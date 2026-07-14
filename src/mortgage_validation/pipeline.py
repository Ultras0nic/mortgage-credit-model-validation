"""One-command reproducible build of data, models, validation evidence, and charts."""

from __future__ import annotations

import json
from pathlib import Path

from .data import run_sql_quality_checks, write_generated_data
from .modeling import fit_models
from .validation import run_validation


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(root: Path | None = None) -> dict:
    root = root or project_root()
    tables = root / "artifacts" / "tables"
    charts = root / "artifacts" / "charts"
    tables.mkdir(parents=True, exist_ok=True)
    df = write_generated_data(root)
    dq = run_sql_quality_checks(df, root / "sql" / "data_quality_checks.sql")
    dq.to_csv(tables / "dq_results.csv", index=False)
    if (dq["status"] != "PASS").any():
        raise RuntimeError("Critical SQL data-quality check failed; see artifacts/tables/dq_results.csv")
    champion, challenger = fit_models(df, root / "artifacts" / "model")
    thresholds = json.loads((root / "config" / "validation_thresholds.json").read_text(encoding="utf-8"))
    summary = run_validation(df, champion, challenger, tables, charts, thresholds)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()

