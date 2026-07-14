import numpy as np

from mortgage_validation.metrics import (
    calibration_intercept_slope,
    ks_statistic,
    population_stability_index,
    wilson_interval,
)


def test_ks_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0


def test_calibration_recovers_identity_on_large_well_calibrated_sample():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.02, 0.30, 50_000)
    y = rng.binomial(1, p)
    intercept, slope = calibration_intercept_slope(y, p)
    assert abs(intercept) < 0.12
    assert 0.90 < slope < 1.10


def test_psi_identical_distribution_is_zero():
    values = __import__("pandas").Series(np.linspace(0, 1, 1_000))
    assert abs(population_stability_index(values, values)) < 1e-12


def test_psi_includes_missingness_shift():
    reference = __import__("pandas").Series([0.0, 1.0] * 50 + [np.nan] * 10)
    comparison = __import__("pandas").Series([0.0, 1.0] * 50 + [np.nan] * 100)
    assert population_stability_index(reference, comparison) > 0.25


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(10, 100)
    assert low < 0.10 < high
