"""PyMC implementation of the hierarchical HTE model from spec §7.

Single coherent posterior over (mu, beta, gamma, tau, alpha). Tier-1
likelihood in v0.1; Tier-2 Poisson terms can be added without changing the
public API.

Engine choice: PyMC 5 (per plan amendment after preflight). Equivalent to the
Stan formulation in spec §7. Diagnostics returned via ArviZ summary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pymc as pm
import arviz as az


@dataclass(frozen=True)
class HTEFitInputs:
    y: np.ndarray            # (N,) log-HR per row
    s: np.ndarray            # (N,) SE per row
    trial: np.ndarray        # (N,) 1..J trial index
    n_trials: int
    X: np.ndarray            # (N, P) covariates


class HTEFit:
    def __init__(self, idata: az.InferenceData, inputs: HTEFitInputs):
        self.idata = idata
        self.inputs = inputs
        self._summary = az.summary(idata, round_to="none")

    def posterior_summary(self, name: str) -> dict[str, float]:
        if name not in self._summary.index:
            raise KeyError(f"{name} not in posterior summary")
        row = self._summary.loc[name]
        return {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "q05": float(row["hdi_3%"]),
            "q95": float(row["hdi_97%"]),
            "rhat": float(row["r_hat"]),
            "ess_bulk": float(row["ess_bulk"]),
        }

    def diagnostics(self) -> dict[str, float]:
        rhat_max = float(self._summary["r_hat"].max())
        ess_min = float(self._summary["ess_bulk"].min())
        divergent = 0
        if hasattr(self.idata, "sample_stats") and "diverging" in self.idata.sample_stats:
            divergent = int(self.idata.sample_stats["diverging"].sum())
        return {"rhat_max": rhat_max, "ess_min": ess_min, "divergent": divergent}

    def draws_dataframe(self):
        return self.idata.posterior.to_dataframe().reset_index()


def fit_hte(inputs: HTEFitInputs, *, iter_warmup: int = 1000,
            iter_sampling: int = 1000, chains: int = 4,
            seed: int = 42) -> HTEFit:
    """Fit the hierarchical HTE model in PyMC.

    Implements spec §7:
        y_i ~ Normal(theta_i, s_i^2)
        theta_i = mu + alpha_trial + X_i . (beta + gamma)
        alpha ~ Normal(0, tau)         (non-centered)
        mu ~ N(0,1); beta ~ N(0,0.5); gamma ~ N(0,0.25); tau ~ HalfNormal(0.5)
    """
    N = int(inputs.y.size)
    P = int(inputs.X.shape[1])
    J = int(inputs.n_trials)
    trial_idx0 = inputs.trial.astype(int) - 1  # PyMC indexing 0..J-1

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 0.5, shape=P)
        gamma = pm.Normal("gamma", 0.0, 0.25, shape=P)
        tau = pm.HalfNormal("tau", 0.5)
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, shape=J)
        alpha = pm.Deterministic("alpha", tau * alpha_raw)

        coef = beta + gamma
        theta = mu + alpha[trial_idx0] + pm.math.dot(inputs.X, coef)
        pm.Normal("y_obs", mu=theta, sigma=inputs.s, observed=inputs.y)

        idata = pm.sample(
            draws=iter_sampling, tune=iter_warmup, chains=chains,
            random_seed=seed, progressbar=False, compute_convergence_checks=False,
            return_inferencedata=True,
        )
    return HTEFit(idata, inputs)
