from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from nur_pce.schema import CovariateKey, CubeCell, UncertaintyDecomp
from nur_pce.output.cube import build_cell, write_cube


def test_build_cell_summarises_posterior():
    rng = np.random.default_rng(0)
    log_hr_draws = rng.normal(-0.30, 0.10, size=2000)
    sampling_var, hte_var, transport_var = 0.005, 0.003, 0.001
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    cell = build_cell(
        key=key, log_hr_draws=log_hr_draws, tier=2,
        var_sampling=sampling_var, var_hte=hte_var, var_transport=transport_var,
    )
    assert isinstance(cell, CubeCell)
    assert 0.7 < cell.hr_mean < 0.78
    assert cell.tier == 2
    assert cell.uncertainty_decomp.total() == pytest.approx(
        sampling_var + hte_var + transport_var, abs=1e-9)


def test_write_cube_round_trips(tmp_path):
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    cell = CubeCell(
        key=key, hr_mean=0.74, hr_credible_95=(0.61, 0.89), p_hr_lt_1=0.99,
        tier=2,
        uncertainty_decomp=UncertaintyDecomp(sampling=0.06, hte=0.04, transport=0.02),
    )
    path = tmp_path / "cube.json"
    write_cube(
        path=path, cells=[cell],
        diagnostics={"rhat_max": 1.004, "ess_min": 1812, "divergent": 0.0},
        drug="finerenone", comparator="placebo", outcome="composite_cardiorenal",
    )
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == "0.1"
    assert loaded["drug"] == "finerenone"
    assert len(loaded["cells"]) == 1
    assert loaded["cells"][0]["tier"] == 2
