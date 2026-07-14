# Independent Model Validation Memo

**Model:** Synthetic Residential Mortgage 12-Month Default Probability Model  
**Validation type:** Initial independent validation case study  
**Validation date:** July 2026  
**Opinion:** **Satisfactory with limitations**

> This educational memo uses synthetic data and public job-description context only. It does not represent Federal Home Loan Bank of Chicago data, models, policies, validation standards, thresholds, risk appetite, or conclusions. All criteria and governance responses are illustrative.

## 1. Executive summary

The validator assessed whether a logistic regression can reliably rank and estimate 12-month mortgage default risk across normal and adverse synthetic conditions. The review covered data quality, conceptual soundness, implementation, discrimination, calibration, backtesting, sensitivity, benchmarking, stability, and governance.

The model separates risk well in the 2022 out-of-time (OOT) sample: AUC is 0.805, Gini is 0.609, and KS is 0.456. All three exceed the illustrative discrimination criteria. However, absolute risk estimation weakens after the synthetic economic and portfolio shift. The observed default rate is 9.08% versus a mean predicted probability of 8.05%. Although the aggregate O/E ratio of 1.128 remains within range, the calibration intercept of 0.345 breaches the ±0.20 criterion. Predicted-PD PSI is 0.338, and several deliberately shifted macro variables also exceed the 0.25 material-drift threshold.

The fixed shallow decision-tree challenger is not a superior alternative. Its OOT AUC is 0.753, and its Brier score is 9.1% worse than the champion's. The paired challenger-minus-champion AUC difference is -0.052 (95% CI -0.067 to -0.036).

The model therefore receives a **Satisfactory with limitations** opinion. The main actions are recalibration on recent representative data and quarterly performance/stability monitoring. Immediate replacement is not recommended.

## 2. Scope and intended use

The case study's business question is:

> Can the model rank and estimate the probability of 12-month default for synthetic residential mortgage originations across baseline and adverse conditions, with sufficient transparency for model-risk monitoring and stakeholder challenge?

`default_12m` represents a synthetic 90+ days-past-due, foreclosure, or charge-off event within 12 months of origination. All model inputs are available at the origination scoring date. The outcome window ends 12 months after origination and is complete by the fixed data-as-of date of December 31, 2023 for every record.

Intended use is limited to educational portfolio-risk ranking and probability estimation. The work is not suitable for lending decisions, pricing, capital, allowance estimates, production scoring, or claims about a real institution. Fair-lending testing, protected-class analysis, decision thresholds, overrides, causal analysis, and production implementation are outside scope; that exclusion is a project boundary, not a judgment that those controls are unnecessary in practice.

## 3. Data and lineage review

The pipeline generates 20,000 mortgage-like loans with fixed seed `20260713`. The portfolio is synthetically weighted toward Illinois and Wisconsin solely to make the case study relevant to the public mission and mortgage focus described in the role context.

| Partition | Cohorts | Loans | Defaults | Default rate |
|---|---|---:|---:|---:|
| Train | 2018–2020 | 11,893 | 530 | 4.46% |
| Validation | 2021 | 4,022 | 202 | 5.02% |
| OOT | 2022 | 4,085 | 371 | 9.08% |

The synthetic data-generating process uses borrower capacity, collateral, loan structure, geography, and economic conditions. Higher FICO and stronger HPI growth reduce default log-odds; higher LTV, DTI, rate spread, unemployment, prior delinquency, cash-out refinance, and investor occupancy increase log-odds. A high-LTV/low-FICO interaction creates a modest nonlinear challenge that is intentionally omitted from the champion.

The 2022 regime includes a stylized change in borrower mix and economic variables. The high macro PSI values demonstrate that the controls detect an adverse shift; they should not be interpreted as estimates of the magnitude of change in any real portfolio.

Seven embedded SQLite checks test duplicate keys, critical and model-input missingness, target validity, numeric ranges, allowed categories, exact outcome-window construction, and maturity by the data-as-of date. All checks pass. The data dictionary and generator serve as the synthetic lineage specification.

## 4. Conceptual soundness and implementation

Logistic regression is reasonable for a binary probability-of-default target and provides an interpretable monotonic log-odds structure. The champion uses median imputation and scaling for numeric features, one-hot encoding for categorical features, and modest L2 regularization. Preprocessing is fit only on 2018–2020 training data. No target, outcome-window, or post-origination field enters the feature set.

Expected directions are supported by the fitted coefficients and sensitivity analysis. Standardized FICO is the largest protective effect; LTV, prior delinquency, unemployment, DTI, and rate spread increase risk. All five one-factor adverse shocks and the combined scenario increase mean PD. The combined scenario is a diagnostic sensitivity exercise, not a causal stress-loss estimate.

The model artifact and preprocessing pipeline are serialized together. Tests verify determinism, schema/ranges, chronological partitioning, SQL control detection, no leakage fields, bounded probabilities, KS behavior, calibration estimation, Wilson intervals, and PSI edge cases.

## 5. Validation results

### 5.1 Discrimination

| Metric | OOT result | Illustrative criterion | Status |
|---|---:|---:|---|
| AUC | 0.805 (95% CI 0.781–0.826) | ≥ 0.70 | Pass |
| Gini | 0.609 | ≥ 0.40 | Pass |
| KS | 0.456 (95% CI 0.420–0.508) | ≥ 0.25 | Pass |
| Validation-to-OOT AUC change | +0.014 | Degradation ≤ 0.05 | Pass |

Rank ordering is stable and suitable for the limited illustrative use.

### 5.2 Calibration and backtesting

| Diagnostic | OOT result | Illustrative criterion | Status |
|---|---:|---:|---|
| Observed / predicted rate | 9.08% / 8.05% | Context | — |
| O/E ratio | 1.128 | 0.80–1.20 | Pass |
| Calibration intercept | 0.345 | Absolute value ≤ 0.20 | **Fail** |
| Calibration slope | 1.096 | 0.80–1.20 | Pass |
| Expected calibration error | 0.013 | ≤ 0.03 | Pass |
| Brier / null Brier | 0.0690 / 0.0826 | Model < null | Pass |
| Adequate-quarter approximate conditional interval coverage | 100% | ≥ 80% | Pass |

The positive calibration intercept indicates systematic underprediction after controlling for the shape of the submitted PDs. The aggregate O/E ratio narrowly passes, so the conclusion is not that all calibration evidence fails; rather, the intercept detects a level shift that motivates recalibration.

### 5.3 Sensitivity

All adverse shocks increase mean predicted PD: FICO -25 (+29.8%), LTV +5 points (+14.3%), DTI +5 points (+8.0%), unemployment +1 point (+28.5%), and rate spread +1 point (+19.6%). The combined diagnostic scenario raises mean PD by 132.6%. Inputs are clipped to documented domains.

### 5.4 Benchmarking

The shallow tree achieves AUC 0.753 versus 0.805 for the champion and Brier score 0.0753 versus 0.0690. Its AUC deficit is statistically clear in the paired bootstrap. This evidence supports retaining the more transparent champion. The challenger is a benchmark, not a production-selection exercise.

### 5.5 Stability

| Driver | Train-to-OOT PSI | Signal |
|---|---:|---|
| FICO | 0.051 | Stable |
| LTV | 0.107 | Watch |
| DTI | 0.058 | Stable |
| Interest rate | 2.528 | Material |
| Unemployment | 1.283 | Material |
| HPI year-over-year change | 2.600 | Material |
| Predicted PD | 0.338 | Material |

PSI is used as a screening signal, not as proof of model failure. Here, score drift aligns with the calibration level shift and higher OOT default rate, so a stability finding is warranted. The deliberately large macro shifts are a synthetic demonstration limitation.

## 6. Findings and recommendations

### F-01 — Moderate: OOT calibration level shift

**Condition.** Calibration intercept is 0.345 against an illustrative ±0.20 limit. O/E is 1.128, slope is 1.096, ECE is 0.013, Brier skill is positive, and approximate conditional quarterly interval coverage is 100%.

**Impact.** Absolute default probability may be understated after a change in economic or portfolio conditions even though rank ordering remains useful.

**Recommendation.** The model owner should recalibrate using recent representative data and validate the change on a later held-out period within 90 days of illustrative acceptance.

**Closure evidence.** Held-out O/E, intercept, slope, ECE, Brier skill, and quarterly coverage meet the configured criteria.

### F-02 — Moderate: material input and score drift

**Condition.** Interest rate, unemployment, HPI change, and predicted PD have PSI above 0.25.

**Impact.** Changing economic and portfolio mix can weaken probability calibration and segment reliability.

**Recommendation.** The model owner should investigate the shifted drivers and implement quarterly PSI, AUC, and O/E monitoring with documented escalation above PSI 0.25.

**Closure evidence.** Root-cause analysis is approved and two consecutive review periods show controlled score stability or an approved limit with effective mitigation.

### F-03 — Observation: challenger does not add value

**Condition.** Challenger-minus-champion AUC is -0.052 and challenger Brier performance is 9.1% worse.

**Recommendation.** Retain the transparent champion. Repeat benchmarking during material redevelopment or when later evidence suggests a stable ≥0.02 AUC or ≥10% Brier improvement.

## 7. Validation opinion

There are no High findings, the implementation and data controls pass, discrimination is strong, and sensitivity directions are economically plausible. Two Moderate findings affect absolute calibration and temporal stability but have bounded impacts and specific mitigations. The illustrative opinion is therefore **Satisfactory with limitations**.

The opinion assumes use remains within the documented educational scope and the calibration/stability actions are tracked. It is not an authorization for production or lending use.

## 8. Monitoring and revalidation triggers

At minimum, an illustrative monitoring pack would report quarterly AUC, KS, observed and predicted default rates, O/E, calibration by risk band, segment counts, feature PSI, and score PSI.

Escalate when:

- score or key-driver PSI exceeds 0.25;
- O/E leaves 0.80–1.20 or calibration misses persist directionally;
- AUC falls below 0.70 or declines by more than 0.05 from the comparison period;
- data-quality controls fail or outcome maturity is incomplete;
- a benchmark shows stable material improvement; or
- model purpose, target, methodology, features, data source, preprocessing, or implementation changes materially.

Any material change should trigger change control, documented impact analysis, and risk-based revalidation before expanded use.

## 9. Limitations

- Data, relationships, outcomes, and adverse conditions are synthetic.
- Macro PSI magnitude is intentionally stylized and is not representative of a real bank portfolio.
- Regularized coefficients are interpretive diagnostics, not classical significance tests.
- Loan-level bootstrap intervals are conditional on the stylized data-generating process and an approximate independent-loan assumption. They do not capture model-form, macro-cluster, data-generation, or macroeconomic uncertainty.
- Unemployment and HPI values are stylized loan-level draws rather than a joined state-month macroeconomic series, so regime detection is demonstrated more strongly than real-world dependence is represented.
- Segment results are diagnostic; small-cell and representativeness limits still apply.
- The tree is a deliberately constrained benchmark, not an exhaustive machine-learning search.
- The project does not assess fair lending, protected classes, pricing, capital, allowance methodology, overrides, operational deployment, or policy thresholds.

## 10. Evidence index and sign-off

Primary evidence is in `artifacts/tables/`: `summary.json`, `acceptance_results.csv`, `findings.csv`, `metrics.csv`, `calibration_deciles.csv`, `backtest_quarterly.csv`, `sensitivity.csv`, `model_comparison.csv`, `psi.csv`, `segment_metrics.csv`, `coefficient_table.csv`, and `dq_results.csv`. Supporting charts are in `artifacts/charts/`. Criteria are versioned in `config/validation_thresholds.json`.

| Role | Name | Date |
|---|---|---|
| Independent validator | Portfolio case-study author | July 2026 |
| Model owner response | Not applicable—synthetic case study | — |
| Model risk approval | Not applicable—synthetic case study | — |
