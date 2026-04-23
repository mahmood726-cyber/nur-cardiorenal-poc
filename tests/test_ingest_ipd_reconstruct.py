from __future__ import annotations
import numpy as np
import pytest
from nur_pce.schema import CovariateKey
from nur_pce.ingest.ipd_reconstruct import (
    reconstruct_ipd, KMPoint, AtRiskInterval,
)


def _profile() -> CovariateKey:
    return CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )


def test_reconstruct_total_n_matches_first_at_risk():
    km = [
        KMPoint(time=0.0, surv=1.000),
        KMPoint(time=0.5, surv=0.950),
        KMPoint(time=1.0, surv=0.890),
        KMPoint(time=1.5, surv=0.830),
    ]
    intervals = [
        AtRiskInterval(t_start=0.0, t_end=0.5, n_at_risk=100),
        AtRiskInterval(t_start=0.5, t_end=1.0, n_at_risk=92),
        AtRiskInterval(t_start=1.0, t_end=1.5, n_at_risk=85),
    ]
    records = reconstruct_ipd(
        km_points=km, at_risk=intervals, arm="treatment",
        trial_id="NCT02540993", profile=_profile(), rng_seed=42,
    )
    assert len(records) == 100
    assert all(r.arm == "treatment" for r in records)
    assert all(r.reconstructed for r in records)


def test_reconstruct_event_count_within_tolerance():
    km = [KMPoint(time=0.0, surv=1.0), KMPoint(time=2.0, surv=0.80)]
    intervals = [AtRiskInterval(t_start=0.0, t_end=2.0, n_at_risk=200)]
    records = reconstruct_ipd(
        km_points=km, at_risk=intervals, arm="control",
        trial_id="NCT02540993", profile=_profile(), rng_seed=7,
    )
    n_events = sum(r.event for r in records)
    expected = 200 * (1 - 0.80)
    assert abs(n_events - expected) <= 4  # within ~10% of expected 40


def test_reconstruct_is_deterministic_under_seed():
    km = [KMPoint(time=0.0, surv=1.0), KMPoint(time=1.0, surv=0.9)]
    intervals = [AtRiskInterval(t_start=0.0, t_end=1.0, n_at_risk=50)]
    a = reconstruct_ipd(km_points=km, at_risk=intervals, arm="treatment",
                        trial_id="NCT02540993", profile=_profile(), rng_seed=99)
    b = reconstruct_ipd(km_points=km, at_risk=intervals, arm="treatment",
                        trial_id="NCT02540993", profile=_profile(), rng_seed=99)
    assert [(r.time, r.event) for r in a] == [(r.time, r.event) for r in b]
