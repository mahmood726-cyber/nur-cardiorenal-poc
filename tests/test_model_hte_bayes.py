from __future__ import annotations
import numpy as np
import pytest
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs


pytestmark = pytest.mark.slow


def _synth_inputs(seed: int = 0) -> HTEFitInputs:
    rng = np.random.default_rng(seed)
    n_trials = 3
    p = 2
    n = 30
    trial = rng.integers(0, n_trials, size=n)
    X = rng.normal(0, 1, size=(n, p))
    true_mu = -0.30
    true_beta = np.array([0.10, -0.05])
    true_gamma = np.array([-0.10, 0.05])
    true_tau = 0.10
    alpha = rng.normal(0, true_tau, size=n_trials)
    theta = true_mu + alpha[trial] + X @ (true_beta + true_gamma)
    s = np.full(n, 0.10)
    y = rng.normal(theta, s)
    return HTEFitInputs(
        y=y, s=s, trial=trial.astype(int) + 1,
        n_trials=n_trials, X=X,
    )


def test_fit_recovers_main_effect_within_tolerance():
    fit = fit_hte(_synth_inputs(seed=1), iter_warmup=500, iter_sampling=500,
                  chains=2, seed=1)
    mu_post = fit.posterior_summary("mu")
    assert abs(mu_post["mean"] - (-0.30)) < 0.10


def test_fit_emits_diagnostics():
    fit = fit_hte(_synth_inputs(seed=2), iter_warmup=500, iter_sampling=500,
                  chains=2, seed=2)
    diag = fit.diagnostics()
    assert "rhat_max" in diag
    assert "ess_min" in diag
    assert "divergent" in diag
