"""Baseline meta-analytic comparators for the held-out validation.

  - Pooled fixed-effect: inverse-variance weighted mean of log-HRs.
  - Meta-regression: weighted least squares on covariates.

These are intentionally simple. The point is to compare PCE against the
*current standard answer*, not the strongest possible competitor.
"""
from __future__ import annotations
import numpy as np


def pooled_fixed_effect(*, log_hrs: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    weights = 1.0 / np.asarray(ses) ** 2
    log_hr = float((np.asarray(log_hrs) * weights).sum() / weights.sum())
    se = float(1.0 / np.sqrt(weights.sum()))
    return {"log_hr": log_hr, "se": se}


def meta_regression(*, log_hrs: np.ndarray, ses: np.ndarray,
                    X: np.ndarray) -> dict[str, float | list[float]]:
    weights = 1.0 / np.asarray(ses) ** 2
    Xd = np.column_stack([np.ones(len(log_hrs)), X])
    W = np.diag(weights)
    XtWX = Xd.T @ W @ Xd
    XtWy = Xd.T @ W @ np.asarray(log_hrs)
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.inv(XtWX)
    return {
        "intercept": float(beta[0]),
        "coefs": [float(b) for b in beta[1:]],
        "se_intercept": float(np.sqrt(cov[0, 0])),
        "se_coefs": [float(np.sqrt(cov[i, i])) for i in range(1, len(beta))],
    }
