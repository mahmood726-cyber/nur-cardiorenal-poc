from __future__ import annotations
import numpy as np
import pytest
from nur_pce.transport.g_formula import project_to_population
from nur_pce.transport.populations import PopulationMarginals


def _toy_population() -> PopulationMarginals:
    return PopulationMarginals(
        region="ToyPop",
        covariate_marginals={
            "x1": {"a": 0.5, "b": 0.5},
            "x2": {"yes": 0.7, "no": 0.3},
        },
        source="test",
    )


def test_projection_returns_per_draw_average():
    cells = {("a", "yes"): -0.30, ("a", "no"): -0.10,
             ("b", "yes"): -0.20, ("b", "no"): 0.05}
    cov_dims = ["x1", "x2"]
    proj = project_to_population(
        cell_log_hrs=cells, covariate_dims=cov_dims, population=_toy_population(),
    )
    expected = (0.5 * 0.7 * -0.30 + 0.5 * 0.3 * -0.10
                + 0.5 * 0.7 * -0.20 + 0.5 * 0.3 * 0.05)
    assert proj == pytest.approx(expected, abs=1e-9)


def test_projection_handles_per_draw_arrays():
    rng = np.random.default_rng(0)
    cells = {("a", "yes"): rng.normal(-0.30, 0.05, size=200),
             ("a", "no"):  rng.normal(-0.10, 0.05, size=200),
             ("b", "yes"): rng.normal(-0.20, 0.05, size=200),
             ("b", "no"):  rng.normal( 0.05, 0.05, size=200)}
    proj = project_to_population(
        cell_log_hrs=cells, covariate_dims=["x1", "x2"], population=_toy_population(),
    )
    assert proj.shape == (200,)
    assert -0.25 < proj.mean() < -0.10
