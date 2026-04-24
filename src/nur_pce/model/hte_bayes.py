"""PyMC implementation of the hierarchical HTE model.

History:
- v0.1.0: spec §7 with both `beta` (prognostic main) and `gamma` (treatment x
  covariate interaction). Tier-1 only.
- v0.1.2: dropped `gamma` (unidentifiable given Tier-1-only data; only
  beta+gamma sum drives the likelihood). Renamed `beta` as the
  treatment-by-covariate interaction.
- v0.2.0: **Tier-2 IPD likelihood added** (closes spec §3a item 1). Per-
  individual reconstructed time-to-event records contribute via an
  **exponential AFT block** sharing the same theta linear predictor.
  Each Tier-2 individual i with arm a_i, time t_i, event d_i contributes:
      eta_i = b0 + alpha_trial[i] + a_i * (mu + X_i . beta)
      log L_i = d_i * eta_i - exp(eta_i) * t_i
  where b0 is a Tier-2-only baseline log-hazard. Note: exponential
  (constant-hazard) is a simplification of the Cox-via-Poisson form spec §7
  named; full discrete-time Cox-via-Poisson is v0.3 work.

Engine: PyMC 5. Diagnostics returned via ArviZ summary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pymc as pm
import arviz as az


@dataclass(frozen=True)
class HTEFitInputs:
    y: np.ndarray            # (N1,) log-HR per Tier-1 row
    s: np.ndarray            # (N1,) SE per Tier-1 row
    trial: np.ndarray        # (N1,) 1..J trial index per Tier-1 row
    n_trials: int
    X: np.ndarray            # (N1, P) Tier-1 covariates
    # Tier-2 IPD (v0.2; all-or-nothing — pass all five or pass none):
    tier2_time: np.ndarray | None = None     # (N2,) time-to-event >= 0
    tier2_event: np.ndarray | None = None    # (N2,) 0/1 event indicator
    tier2_arm: np.ndarray | None = None      # (N2,) 0=control, 1=treatment
    tier2_X: np.ndarray | None = None        # (N2, P) Tier-2 covariates
    tier2_trial: np.ndarray | None = None    # (N2,) 1..J trial index per Tier-2 row

    @property
    def has_tier2(self) -> bool:
        return self.tier2_time is not None


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
            seed: int = 42, target_accept: float = 0.8) -> HTEFit:
    """Fit the hierarchical HTE model in PyMC.

    Tier-1-only formulation:
        y_i ~ Normal(theta_i, s_i^2)
        theta_i = mu + alpha_trial + X_i . beta
        alpha ~ Normal(0, tau)         (non-centered)
        mu ~ N(0,1); beta ~ N(0,0.5); tau ~ HalfNormal(0.5)

    `beta` is the treatment-by-covariate interaction (moderation of log-HR by X).
    The composite coefficient inherited from the v0.1 spec §7 (`beta + gamma`)
    is reported here as the single identifiable `beta`. See module docstring.
    """
    N = int(inputs.y.size)
    P = int(inputs.X.shape[1])
    J = int(inputs.n_trials)
    trial_idx0 = inputs.trial.astype(int) - 1  # PyMC indexing 0..J-1

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 0.5, shape=P)
        tau = pm.HalfNormal("tau", 0.5)
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, shape=J)
        alpha = pm.Deterministic("alpha", tau * alpha_raw)

        # Tier-1 likelihood (subgroup log-HRs)
        theta = mu + alpha[trial_idx0] + pm.math.dot(inputs.X, beta)
        pm.Normal("y_obs", mu=theta, sigma=inputs.s, observed=inputs.y)

        # Tier-2 likelihood (per-individual exponential AFT, v0.2).
        #
        # Design choice: alpha (trial RE) is *Tier-1-specific* — it captures
        # trial-level variation in published subgroup HRs (publication-window
        # effects, slightly different cohort framings, etc.). Tier-2
        # individual-level records see the *underlying* treatment effect
        # directly, with no trial-specific shift. So alpha does not appear
        # in Tier-2's eta. mu and beta are shared.
        #
        # b0 is a global Tier-2 baseline log-hazard. Trial-level baseline
        # variation in Tier-2 is not modelled (could be added as a separate
        # b0_trial random effect in a future iteration).
        if inputs.has_tier2:
            b0 = pm.Normal("b0", 0.0, 1.0)
            treatment_effect = mu + pm.math.dot(inputs.tier2_X, beta)
            eta_t2 = b0 + inputs.tier2_arm * treatment_effect
            log_lik = (
                inputs.tier2_event * eta_t2
                - pm.math.exp(eta_t2) * inputs.tier2_time
            )
            pm.Potential("tier2_likelihood", pm.math.sum(log_lik))

        idata = pm.sample(
            draws=iter_sampling, tune=iter_warmup, chains=chains,
            random_seed=seed, progressbar=False, compute_convergence_checks=False,
            return_inferencedata=True, target_accept=target_accept,
        )
    return HTEFit(idata, inputs)


def tier2_records_to_arrays(records, trial_to_idx: dict[str, int],
                            x_encoder) -> dict[str, np.ndarray]:
    """Convert list[Tier2Record] to the array form HTEFitInputs expects.

    Args:
        records: list of Tier2Record
        trial_to_idx: maps trial_id -> 1..J integer
        x_encoder: callable Tier2Record -> 1-D numpy array of length P
    """
    n = len(records)
    arm = np.array([1 if r.arm == "treatment" else 0 for r in records], dtype=int)
    time = np.array([r.time for r in records], dtype=float)
    event = np.array([r.event for r in records], dtype=int)
    trial = np.array([trial_to_idx[r.trial_id] for r in records], dtype=int)
    X = np.stack([x_encoder(r) for r in records]) if n else np.zeros((0, 0))
    return {"tier2_time": time, "tier2_event": event, "tier2_arm": arm,
            "tier2_X": X, "tier2_trial": trial}
