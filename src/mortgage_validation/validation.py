"""Independent validation analyses and evidence generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .charts import backtest_chart, bar_chart, calibration_chart, comparison_chart, discrimination_chart
from .metrics import (
    bootstrap_interval,
    calibration_table,
    ks_statistic,
    metric_row,
    paired_bootstrap_difference,
    population_stability_index,
    wilson_interval,
)
from .modeling import FEATURES, NUMERIC_FEATURES, TARGET, coefficient_table


def _finding(
    fid: str,
    category: str,
    severity: str,
    criterion: str,
    condition: str,
    evidence: str,
    impact: str,
    recommendation: str,
    closure_evidence: str,
) -> dict[str, str]:
    observation = severity == "Observation"
    return {
        "finding_id": fid,
        "category": category,
        "severity": severity,
        "criterion": criterion,
        "condition": condition,
        "evidence": evidence,
        "impact": impact,
        "recommendation": recommendation,
        "owner": "Model Risk Management" if observation else "Model Owner",
        "target_date": "Next periodic benchmark review" if observation else "Within 90 days of illustrative acceptance",
        "status": "Noted" if observation else "Open",
        "closure_evidence": closure_evidence,
    }


def _acceptance_row(
    criterion_id: str,
    area: str,
    criterion: str,
    observed: str,
    threshold: str,
    passed: bool,
    evidence: str,
) -> dict[str, str]:
    return {
        "criterion_id": criterion_id,
        "area": area,
        "criterion": criterion,
        "observed": observed,
        "threshold": threshold,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def run_validation(df: pd.DataFrame, champion, challenger, output_dir: Path, chart_dir: Path, thresholds: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for split in ["train", "validation", "oot"]:
        sample = df.loc[df["split"] == split]
        pred = champion.predict_proba(sample[FEATURES])[:, 1]
        predictions[split] = pred
        row = metric_row(sample[TARGET].to_numpy(), pred, split)
        if split == "oot":
            row["auc_ci_low"], row["auc_ci_high"] = bootstrap_interval(sample[TARGET].to_numpy(), pred, roc_auc_score)
            row["ks_ci_low"], row["ks_ci_high"] = bootstrap_interval(sample[TARGET].to_numpy(), pred, ks_statistic)
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    oot = df.loc[df["split"] == "oot"].copy()
    oot_pred = predictions["oot"]
    challenger_pred = challenger.predict_proba(oot[FEATURES])[:, 1]
    cal = calibration_table(oot[TARGET], oot_pred)
    calibration_ece = float(
        np.average(
            np.abs(cal["observed_rate"] - cal["predicted_rate"]),
            weights=cal["n"],
        )
    )
    cal.to_csv(output_dir / "calibration_deciles.csv", index=False)

    oot["predicted_pd"] = oot_pred
    oot["prediction_variance"] = oot_pred * (1 - oot_pred)
    oot["quarter"] = pd.to_datetime(oot["origination_date"]).dt.to_period("Q").astype(str)
    backtest = oot.groupby("quarter", as_index=False).agg(
        n=(TARGET, "size"),
        defaults=(TARGET, "sum"),
        observed_rate=(TARGET, "mean"),
        predicted_rate=("predicted_pd", "mean"),
        expected_defaults=("predicted_pd", "sum"),
        expected_variance=("prediction_variance", "sum"),
    )
    backtest["oe_ratio"] = backtest["defaults"] / backtest["expected_defaults"]
    expected_std = np.sqrt(backtest["expected_variance"])
    backtest["lower_95_defaults"] = np.maximum(0, backtest["expected_defaults"] - 1.96 * expected_std)
    backtest["upper_95_defaults"] = np.minimum(
        backtest["n"], backtest["expected_defaults"] + 1.96 * expected_std
    )
    backtest["within_95_interval"] = backtest["defaults"].between(
        backtest["lower_95_defaults"], backtest["upper_95_defaults"]
    )
    backtest["adequate_count"] = (
        backtest["expected_defaults"] >= thresholds["backtest_min_expected_defaults"]
    )
    observed_intervals = [wilson_interval(int(row.defaults), int(row.n)) for row in backtest.itertuples()]
    backtest["observed_rate_ci_low"] = [interval[0] for interval in observed_intervals]
    backtest["observed_rate_ci_high"] = [interval[1] for interval in observed_intervals]
    adequate_backtest = backtest.loc[backtest["adequate_count"]]
    backtest_coverage = (
        float(adequate_backtest["within_95_interval"].mean())
        if not adequate_backtest.empty
        else float("nan")
    )
    backtest.to_csv(output_dir / "backtest_quarterly.csv", index=False)

    segment_rows = []
    segment_definitions = {
        "lower_fico": oot["fico_score"] < 680,
        "high_ltv": oot["ltv"] > 95,
        "high_dti": oot["dti"] > 45,
        "IL": oot["state"] == "IL",
        "WI": oot["state"] == "WI",
    }
    for segment, mask in segment_definitions.items():
        if mask.sum() > 0:
            row = metric_row(oot.loc[mask, TARGET].to_numpy(), oot.loc[mask, "predicted_pd"].to_numpy(), segment)
            row["segment"] = segment
            segment_rows.append(row)
    pd.DataFrame(segment_rows).to_csv(output_dir / "segment_metrics.csv", index=False)

    base_pd = float(oot_pred.mean())
    shocks = {"fico_score": -25.0, "ltv": 5.0, "dti": 5.0, "unemployment_rate": 1.0, "rate_spread": 1.0}
    bounds = {
        "fico_score": (300, 850),
        "ltv": (20, 150),
        "dti": (0, 80),
        "unemployment_rate": (2, 15),
        "rate_spread": (-1, 10),
    }
    sensitivity_rows = []
    combined = oot[FEATURES].copy()
    for feature, shock in shocks.items():
        stressed = oot[FEATURES].copy()
        stressed[feature] = (stressed[feature] + shock).clip(*bounds[feature])
        stressed_pd = champion.predict_proba(stressed)[:, 1]
        sensitivity_rows.append({"scenario": f"{feature} {shock:+g}", "mean_pd": stressed_pd.mean(), "absolute_change": stressed_pd.mean() - base_pd, "relative_change": stressed_pd.mean() / base_pd - 1, "expected_direction": "increase", "direction_pass": stressed_pd.mean() > base_pd})
        combined[feature] = (combined[feature] + shock).clip(*bounds[feature])
    combined_pd = champion.predict_proba(combined)[:, 1]
    sensitivity_rows.append({"scenario": "combined_adverse", "mean_pd": combined_pd.mean(), "absolute_change": combined_pd.mean() - base_pd, "relative_change": combined_pd.mean() / base_pd - 1, "expected_direction": "increase", "direction_pass": combined_pd.mean() > base_pd})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)

    train = df.loc[df["split"] == "train"]
    psi_rows = []
    for feature in ["fico_score", "ltv", "dti", "interest_rate", "unemployment_rate", "hpi_yoy_change"]:
        psi_rows.append({"feature": feature, "psi": population_stability_index(train[feature], oot[feature])})
    psi_rows.append({"feature": "predicted_pd", "psi": population_stability_index(pd.Series(predictions["train"]), pd.Series(oot_pred))})
    psi = pd.DataFrame(psi_rows)
    psi["signal"] = np.select([psi["psi"] > thresholds["psi_material"], psi["psi"] >= thresholds["psi_watch"]], ["material", "watch"], default="stable")
    psi.to_csv(output_dir / "psi.csv", index=False)

    comparison = pd.DataFrame([
        {"model": "Champion logistic", **metric_row(oot[TARGET].to_numpy(), oot_pred, "oot")},
        {"model": "Challenger tree", **metric_row(oot[TARGET].to_numpy(), challenger_pred, "oot")},
    ])
    auc_delta_ci_low, auc_delta_ci_high = paired_bootstrap_difference(
        oot[TARGET].to_numpy(), oot_pred, challenger_pred, roc_auc_score
    )
    comparison["auc_delta_vs_champion"] = [0.0, comparison.iloc[1]["auc"] - comparison.iloc[0]["auc"]]
    comparison["auc_delta_ci_low"] = [np.nan, auc_delta_ci_low]
    comparison["auc_delta_ci_high"] = [np.nan, auc_delta_ci_high]
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    coefficient_table(champion).to_csv(output_dir / "coefficient_table.csv", index=False)

    oot_metrics = metrics.loc[metrics["split"] == "oot"].iloc[0]
    validation_metrics = metrics.loc[metrics["split"] == "validation"].iloc[0]
    auc_degradation = float(validation_metrics["auc"] - oot_metrics["auc"])
    material = psi.loc[psi["psi"] > thresholds["psi_material"], "feature"].tolist()
    score_psi = float(psi.loc[psi["feature"] == "predicted_pd", "psi"].iloc[0])
    auc_delta = float(comparison.iloc[1]["auc"] - comparison.iloc[0]["auc"])
    brier_improvement = float(
        (comparison.iloc[0]["brier"] - comparison.iloc[1]["brier"])
        / comparison.iloc[0]["brier"]
    )
    benchmark_material = (
        auc_delta >= thresholds["challenger_auc_delta_material"]
        or brier_improvement >= thresholds["challenger_brier_improvement_material"]
    )
    dq_path = output_dir / "dq_results.csv"
    dq = pd.read_csv(dq_path) if dq_path.exists() else pd.DataFrame()
    dq_pass = not dq.empty and bool((dq["status"] == "PASS").all())
    dq_observed = f"{int((dq['status'] == 'PASS').sum())}/{len(dq)} PASS" if not dq.empty else "not available"

    acceptance_rows = [
        _acceptance_row("DQ-01", "Data quality", "All critical SQL checks pass", dq_observed, "All PASS", dq_pass, "dq_results.csv"),
        _acceptance_row("SMP-01", "Sample sufficiency", "OOT sample has sufficient defaults", f"{int(oot_metrics['defaults'])}", f">= {thresholds['oot_defaults_min']}", oot_metrics["defaults"] >= thresholds["oot_defaults_min"], "metrics.csv: oot defaults"),
        _acceptance_row("DIS-01", "Discrimination", "OOT AUC meets minimum", f"{oot_metrics['auc']:.4f}", f">= {thresholds['auc_min']:.2f}", oot_metrics["auc"] >= thresholds["auc_min"], "metrics.csv: oot auc"),
        _acceptance_row("DIS-02", "Discrimination", "OOT Gini meets minimum", f"{oot_metrics['gini']:.4f}", f">= {thresholds['gini_min']:.2f}", oot_metrics["gini"] >= thresholds["gini_min"], "metrics.csv: oot gini"),
        _acceptance_row("DIS-03", "Discrimination", "OOT KS meets minimum", f"{oot_metrics['ks']:.4f}", f">= {thresholds['ks_min']:.2f}", oot_metrics["ks"] >= thresholds["ks_min"], "metrics.csv: oot ks"),
        _acceptance_row("DIS-04", "Discrimination", "Validation-to-OOT AUC degradation is within limit", f"{auc_degradation:.4f}", f"<= {thresholds['auc_degradation_max']:.2f}", auc_degradation <= thresholds["auc_degradation_max"], "metrics.csv: validation and oot auc"),
        _acceptance_row("CAL-01", "Calibration", "Brier score beats prevalence-only benchmark", f"{oot_metrics['brier']:.4f} vs {oot_metrics['brier_null']:.4f}", "Brier < null Brier", oot_metrics["brier"] < oot_metrics["brier_null"], "metrics.csv: oot brier and brier_null"),
        _acceptance_row("CAL-02", "Calibration", "Overall observed/expected ratio is in range", f"{oot_metrics['oe_ratio']:.4f}", f"{thresholds['oe_min']:.2f} to {thresholds['oe_max']:.2f}", thresholds["oe_min"] <= oot_metrics["oe_ratio"] <= thresholds["oe_max"], "metrics.csv: oot oe_ratio"),
        _acceptance_row("CAL-03", "Calibration", "Absolute calibration intercept is within limit", f"{oot_metrics['calibration_intercept']:.4f}", f"abs <= {thresholds['calibration_intercept_abs_max']:.2f}", abs(oot_metrics["calibration_intercept"]) <= thresholds["calibration_intercept_abs_max"], "metrics.csv: oot calibration_intercept"),
        _acceptance_row("CAL-04", "Calibration", "Calibration slope is in range", f"{oot_metrics['calibration_slope']:.4f}", f"{thresholds['calibration_slope_min']:.2f} to {thresholds['calibration_slope_max']:.2f}", thresholds["calibration_slope_min"] <= oot_metrics["calibration_slope"] <= thresholds["calibration_slope_max"], "metrics.csv: oot calibration_slope"),
        _acceptance_row("CAL-05", "Calibration", "Expected calibration error is within limit", f"{calibration_ece:.4f}", f"<= {thresholds['ece_max']:.2f}", calibration_ece <= thresholds["ece_max"], "calibration_deciles.csv"),
        _acceptance_row("BT-01", "Backtesting", "Adequate quarterly buckets fall within approximate conditional 95% expected-default intervals", f"{backtest_coverage:.1%}" if np.isfinite(backtest_coverage) else "no adequate buckets", f">= {thresholds['backtest_interval_coverage_min']:.0%}", np.isfinite(backtest_coverage) and backtest_coverage >= thresholds["backtest_interval_coverage_min"], "backtest_quarterly.csv"),
        _acceptance_row("SEN-01", "Sensitivity", "All documented adverse shocks increase mean PD", f"{int(sensitivity['direction_pass'].sum())}/{len(sensitivity)} pass", "All pass", bool(sensitivity["direction_pass"].all()), "sensitivity.csv"),
        _acceptance_row("BMK-01", "Benchmarking", "Challenger does not materially outperform champion", f"AUC delta {auc_delta:.4f}; Brier improvement {brier_improvement:.1%}", f"AUC delta < {thresholds['challenger_auc_delta_material']:.2f} and Brier improvement < {thresholds['challenger_brier_improvement_material']:.0%}", not benchmark_material, "model_comparison.csv"),
        _acceptance_row("DRF-01", "Stability", "Score PSI is below material threshold", f"{score_psi:.4f}", f"<= {thresholds['psi_material']:.2f}", score_psi <= thresholds["psi_material"], "psi.csv: predicted_pd"),
        _acceptance_row("DRF-02", "Stability", "No monitored feature has material PSI", f"{len(material)} material", "0 material", len(material) == 0, "psi.csv"),
    ]
    acceptance = pd.DataFrame(acceptance_rows)
    acceptance.to_csv(output_dir / "acceptance_results.csv", index=False)

    findings: list[dict[str, str]] = []
    calibration_failed = (
        oot_metrics["oe_ratio"] < thresholds["oe_min"]
        or oot_metrics["oe_ratio"] > thresholds["oe_max"]
        or abs(oot_metrics["calibration_intercept"]) > thresholds["calibration_intercept_abs_max"]
        or oot_metrics["calibration_slope"] < thresholds["calibration_slope_min"]
        or oot_metrics["calibration_slope"] > thresholds["calibration_slope_max"]
        or oot_metrics["brier"] >= oot_metrics["brier_null"]
        or calibration_ece > thresholds["ece_max"]
        or not np.isfinite(backtest_coverage)
        or backtest_coverage < thresholds["backtest_interval_coverage_min"]
    )
    if calibration_failed:
        findings.append(_finding(
            "F-01", "Performance/Calibration", "Moderate",
            "CAL-01 through CAL-05 and BT-01 in acceptance_results.csv",
            f"OOT O/E={oot_metrics['oe_ratio']:.3f}, intercept={oot_metrics['calibration_intercept']:.3f}, slope={oot_metrics['calibration_slope']:.3f}, ECE={calibration_ece:.3f}, Brier={oot_metrics['brier']:.4f} vs null={oot_metrics['brier_null']:.4f}, and quarterly interval coverage={backtest_coverage:.0%}.",
            "metrics.csv, calibration_deciles.csv, backtest_quarterly.csv, and acceptance_results.csv",
            "Absolute default risk may be misstated under shifted conditions even when rank ordering remains useful.",
            "Recalibrate on recent representative data and confirm performance on a later held-out period.",
            "Held-out O/E, intercept, slope, ECE, Brier skill, and quarterly interval coverage all meet the configured criteria.",
        ))
    if material:
        material_values = "; ".join(
            f"{row.feature}={row.psi:.3f}" for row in psi.loc[psi["feature"].isin(material)].itertuples()
        )
        findings.append(_finding(
            "F-02", "Stability/Drift", "Moderate", "DRF-01 and DRF-02 in acceptance_results.csv",
            f"PSI exceeds {thresholds['psi_material']:.2f}: {material_values}.",
            "psi.csv and acceptance_results.csv",
            "Changing population or economic mix can weaken calibration and segment reliability.",
            f"Investigate the identified drivers and implement quarterly PSI/AUC/OE monitoring with escalation when PSI exceeds {thresholds['psi_material']:.2f}.",
            "Root-cause analysis, monitoring evidence, and two consecutive review periods showing controlled score stability or approved limits.",
        ))
    findings.append(_finding(
        "F-03", "Benchmarking", "Moderate" if benchmark_material else "Observation", "BMK-01 in acceptance_results.csv",
        f"Challenger-minus-champion OOT AUC={auc_delta:.3f} (paired 95% CI {auc_delta_ci_low:.3f} to {auc_delta_ci_high:.3f}); relative Brier improvement={brier_improvement:.1%}.",
        "model_comparison.csv and acceptance_results.csv",
        "A benchmark can reveal nonlinear performance opportunity but introduces explainability trade-offs.",
        "Assess redevelopment if material gains persist; otherwise retain the transparent champion and repeat benchmarking at redevelopment.",
        "Documented benchmark comparison on a later OOT sample using the same population, features, and metrics.",
    ))
    if not sensitivity["direction_pass"].all():
        failed_scenarios = ", ".join(sensitivity.loc[~sensitivity["direction_pass"], "scenario"])
        findings.append(_finding(
            "F-04", "Conceptual Soundness", "Moderate", "SEN-01 in acceptance_results.csv",
            f"Adverse scenarios without an increase in mean PD: {failed_scenarios}.",
            "sensitivity.csv and acceptance_results.csv",
            "Unexpected response may conflict with economic intuition and undermine use in stress analysis.",
            "Review transformations and coefficient direction for the failed scenarios.",
            "Re-run sensitivity evidence showing all core shocks behave as documented or an approved theoretical rationale for exceptions.",
        ))
    discrimination_failed = (
        oot_metrics["auc"] < thresholds["auc_min"]
        or oot_metrics["gini"] < thresholds["gini_min"]
        or oot_metrics["ks"] < thresholds["ks_min"]
        or auc_degradation > thresholds["auc_degradation_max"]
    )
    if discrimination_failed:
        discrimination_severity = (
            "High"
            if oot_metrics["auc"] < thresholds["auc_unsatisfactory_below"]
            or oot_metrics["ks"] < thresholds["ks_unsatisfactory_below"]
            else "Moderate"
        )
        findings.append(_finding(
            "F-05", "Performance/Discrimination", discrimination_severity,
            "DIS-01 through DIS-03 in acceptance_results.csv",
            f"OOT AUC={oot_metrics['auc']:.3f}, Gini={oot_metrics['gini']:.3f}, KS={oot_metrics['ks']:.3f}, and validation-to-OOT AUC degradation={auc_degradation:.3f}.",
            "metrics.csv and acceptance_results.csv",
            "Weak rank ordering limits the model's ability to distinguish higher- from lower-risk loans.",
            "Reassess variable design and model specification, then validate discrimination on untouched OOT data.",
            "OOT AUC, Gini, and KS meet the configured criteria with uncertainty discussed.",
        ))
    if oot_metrics["defaults"] < thresholds["oot_defaults_min"]:
        findings.append(_finding(
            "F-06", "Performance/Sample Sufficiency", "Moderate", "SMP-01 in acceptance_results.csv",
            f"OOT defaults={int(oot_metrics['defaults'])}; minimum={thresholds['oot_defaults_min']}.",
            "metrics.csv and acceptance_results.csv",
            "Limited event counts widen uncertainty and reduce the reliability of headline and segment conclusions.",
            "Accumulate additional OOT performance data before relying on fine-grained validation conclusions.",
            f"An OOT sample with at least {thresholds['oot_defaults_min']} defaults and updated confidence intervals.",
        ))
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(output_dir / "findings.csv", index=False)

    discrimination_chart(oot[TARGET].to_numpy(), oot_pred, challenger_pred, chart_dir / "01_discrimination.png")
    calibration_chart(cal, chart_dir / "02_calibration.png")
    backtest_chart(backtest, chart_dir / "03_backtest.png")
    bar_chart(psi, "feature", "psi", "Train-to-OOT population stability", chart_dir / "04_psi.png", thresholds["psi_material"])
    bar_chart(sensitivity, "scenario", "relative_change", "Adverse sensitivity: mean PD change", chart_dir / "05_sensitivity.png")
    comparison_chart(comparison, chart_dir / "06_champion_vs_challenger.png")

    if (findings_df["severity"] == "High").any():
        conclusion = "Unsatisfactory"
    elif (findings_df["severity"] == "Moderate").any():
        conclusion = "Satisfactory with limitations"
    else:
        conclusion = "Satisfactory"
    summary = {
        "conclusion": conclusion,
        "oot_auc": float(oot_metrics["auc"]),
        "oot_ks": float(oot_metrics["ks"]),
        "oot_observed_rate": float(oot_metrics["observed_rate"]),
        "oot_predicted_rate": float(oot_metrics["predicted_rate"]),
        "oot_oe_ratio": float(oot_metrics["oe_ratio"]),
        "oot_ece": calibration_ece,
        "backtest_interval_coverage": backtest_coverage,
        "material_psi_features": material,
        "acceptance_failures": acceptance.loc[acceptance["status"] == "FAIL", "criterion_id"].tolist(),
        "high_findings": int((findings_df["severity"] == "High").sum()),
        "moderate_findings": int((findings_df["severity"] == "Moderate").sum()),
        "disclaimer": thresholds["disclaimer"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
