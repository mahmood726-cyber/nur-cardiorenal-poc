"""G-formula posterior projection to a target population.

For each covariate cell c with log-HR posterior theta_c (scalar or per-draw
array), the population-projected log-HR is

    log( sum_c P_T(c) * exp(theta_c) )

i.e. the average is taken on the natural HR scale (per spec §7), then
log-transformed back. This is the unbiased E_{P_T(X)}[exp(theta(X))]
formulation; averaging on the log-HR scale (the v0.1 implementation)
introduces a Jensen-bias proportional to the per-draw variance.

v0.1.3 (2026-04-23, post-final-review): switched from log-scale averaging
to natural-scale averaging. Closes spec §3a item 3.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence
from nur_pce.transport.populations import PopulationMarginals


def project_to_population(
    *,
    cell_log_hrs: dict[tuple, np.ndarray | float],
    covariate_dims: Sequence[str],
    population: PopulationMarginals,
) -> np.ndarray | float:
    """Population-projected log-HR via natural-scale weighted average.

    Returns log( sum_c P_T(c) * exp(theta_c) ) per draw (or scalar).
    """
    accum: np.ndarray | float = 0.0
    for cell_key, log_hr in cell_log_hrs.items():
        if len(cell_key) != len(covariate_dims):
            raise ValueError(
                f"cell key {cell_key} length != covariate_dims {covariate_dims}"
            )
        weight = 1.0
        for cov_name, level in zip(covariate_dims, cell_key):
            level_str = str(level).lower() if isinstance(level, bool) else str(level)
            dist = population.covariate_marginals[cov_name]
            weight *= dist[level_str if level_str in dist else str(level)]
        accum = accum + weight * np.exp(np.asarray(log_hr))
    return np.log(accum)
