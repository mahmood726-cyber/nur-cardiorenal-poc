from __future__ import annotations
from pathlib import Path
import pytest
from nur_pce.transport.populations import load_populations, PopulationLoadError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "populations_synth.json"


def test_load_populations_returns_two_targets():
    pops = load_populations(FIXTURE)
    assert set(pops.keys()) == {"USA", "Pakistan"}


def test_marginals_sum_to_one_per_covariate():
    pops = load_populations(FIXTURE)
    for region, p in pops.items():
        for cov, dist in p.covariate_marginals.items():
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-6, f"{region}/{cov} sums to {total}"


def test_marginals_normalised_if_off_by_small_amount(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"populations": {"X": {"covariate_marginals": '
                   '{"sex": {"M": 0.6, "F": 0.41}}, "source": "s"}}}')
    pops = load_populations(bad)
    sex = pops["X"].covariate_marginals["sex"]
    assert abs(sum(sex.values()) - 1.0) < 1e-9


def test_load_fails_closed_on_grossly_wrong_marginals(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"populations": {"X": {"covariate_marginals": '
                   '{"sex": {"M": 0.1, "F": 0.1}}, "source": "s"}}}')
    with pytest.raises(PopulationLoadError):
        load_populations(bad)
