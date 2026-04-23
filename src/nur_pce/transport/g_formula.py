"""G-formula posterior projection to a target population.

For each covariate cell c with log-HR posterior theta_c (scalar or per-draw
array), the population-projected log-HR is sum_c P_T(c) * theta_c.
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
    """Weighted average of per-cell log-HR by P_T(cell)."""
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
        accum = accum + weight * np.asarray(log_hr)
    return accum
