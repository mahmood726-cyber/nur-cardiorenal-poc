"""Test that Tier-2 IPD contribution shrinks the posterior on the treatment
effect parameter mu — proving the wire-through is real, not cosmetic.

v0.2 spec §3a item 1.
"""
from __future__ import annotations
import numpy as np
import pytest
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs


pytestmark = pytest.mark.slow


def _synth_tier1(seed: int, true_mu: float = -0.30,
                  s_value: float = 0.30) -> HTEFitInputs:
    """Synthetic Tier-1: 6 subgroup rows from 2 trials.

    Default SE = 0.30 (relatively weak) so Tier-2's ~120-event contribution
    can move the joint posterior measurably tighter than Tier-1 alone.
    """
    rng = np.random.default_rng(seed)
    n_trials = 2
    p = 2
    n = 6
    trial = rng.integers(0, n_trials, size=n)
    X = rng.normal(0, 1, size=(n, p))
    true_beta = np.array([0.0, 0.0])
    true_tau = 0.05
    alpha = rng.normal(0, true_tau, size=n_trials)
    theta = true_mu + alpha[trial] + X @ true_beta
    s = np.full(n, s_value)
    y = rng.normal(theta, s)
    return HTEFitInputs(
        y=y, s=s, trial=trial.astype(int) + 1,
        n_trials=n_trials, X=X,
    )


def _synth_tier1_plus_tier2(seed: int, true_mu: float = -0.30,
                             n_t2: int = 200) -> HTEFitInputs:
    """Same Tier-1 plus Tier-2 IPD generated under the same true mu.

    For each Tier-2 individual:
      arm ~ Bernoulli(0.5)
      time ~ Exponential(rate = exp(arm * true_mu))    [b0 absorbed = 0]
      event with prob 0.6, else administrative censoring
    """
    base = _synth_tier1(seed=seed, true_mu=true_mu)
    rng = np.random.default_rng(seed + 1000)
    arm = rng.integers(0, 2, size=n_t2)
    rate = np.exp(arm * true_mu)
    raw_time = rng.exponential(scale=1.0 / rate, size=n_t2)
    cens_time = rng.exponential(scale=2.0, size=n_t2)
    time = np.minimum(raw_time, cens_time)
    event = (raw_time <= cens_time).astype(int)
    trial = rng.integers(0, base.n_trials, size=n_t2) + 1
    X_t2 = rng.normal(0, 1, size=(n_t2, base.X.shape[1]))
    return HTEFitInputs(
        y=base.y, s=base.s, trial=base.trial,
        n_trials=base.n_trials, X=base.X,
        tier2_time=time, tier2_event=event, tier2_arm=arm,
        tier2_X=X_t2, tier2_trial=trial,
    )


def test_tier2_tightens_mu_posterior():
    """Adding Tier-2 IPD to the fit must reduce mu's posterior SD AND
    keep the posterior centred near true mu — proves Tier-2 is contributing
    real information, not noise.
    """
    fit_t1 = fit_hte(_synth_tier1(seed=1),
                      iter_warmup=400, iter_sampling=400, chains=2,
                      seed=1, target_accept=0.95)
    mu_t1 = fit_t1.posterior_summary("mu")
    sd_t1 = mu_t1["sd"]

    fit_t12 = fit_hte(_synth_tier1_plus_tier2(seed=1),
                       iter_warmup=400, iter_sampling=400, chains=2,
                       seed=1, target_accept=0.95)
    mu_t12 = fit_t12.posterior_summary("mu")
    sd_t12 = mu_t12["sd"]

    assert sd_t12 < sd_t1, (
        f"Tier-2 should tighten mu posterior; got SD t1={sd_t1:.4f} "
        f"vs t1+t2={sd_t12:.4f}"
    )
    # The posterior mean should be in the ballpark of true -0.30. Tolerance
    # is generous (0.40) because with n_trials=2 + weak Tier-1 SE=0.30, the
    # mu-vs-alpha aliasing is strong on this synthetic fixture; tight
    # recovery is a real-data concern, not a synthetic-POC concern.
    assert abs(mu_t12["mean"] - (-0.30)) < 0.40, (
        f"with Tier-2, mu mean={mu_t12['mean']:.3f}, expected loosely near -0.30"
    )
