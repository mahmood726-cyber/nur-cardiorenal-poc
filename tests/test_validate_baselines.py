from __future__ import annotations
import numpy as np
import pytest
from nur_pce.validate.baselines import pooled_fixed_effect, meta_regression


def test_pooled_fe_matches_inverse_variance():
    log_hrs = np.array([-0.30, -0.40, -0.20])
    ses = np.array([0.10, 0.15, 0.12])
    res = pooled_fixed_effect(log_hrs=log_hrs, ses=ses)
    weights = 1.0 / ses ** 2
    expected = (log_hrs * weights).sum() / weights.sum()
    assert res["log_hr"] == pytest.approx(expected)
    assert res["se"] == pytest.approx(1.0 / np.sqrt(weights.sum()))


def test_meta_regression_recovers_slope():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(0, 1, size=n)
    true_intercept = -0.30
    true_slope = -0.15
    log_hrs = true_intercept + true_slope * x + rng.normal(0, 0.05, size=n)
    ses = np.full(n, 0.10)
    res = meta_regression(log_hrs=log_hrs, ses=ses, X=x.reshape(-1, 1))
    assert abs(res["intercept"] - true_intercept) < 0.10
    assert abs(res["coefs"][0] - true_slope) < 0.10
