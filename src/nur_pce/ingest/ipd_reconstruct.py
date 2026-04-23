"""Tier-2 IPD reconstruction via the Guyot et al. 2012 algorithm.

Inputs:
  - Digitised KM curve points (time, surv).
  - Numbers-at-risk per interval (typically reported in the trial's KM figure).

Outputs:
  - Reconstructed individual time-to-event records (Tier2Record), flagged as
    reconstructed=True.

References:
  Guyot P, Ades AE, Ouwens MJNM, Welton NJ. Enhanced secondary analysis of
  survival data: reconstructing the data from published Kaplan-Meier survival
  curves. BMC Med Res Methodol 2012; 12:9.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from nur_pce.schema import CovariateKey, Tier2Record


@dataclass(frozen=True)
class KMPoint:
    time: float
    surv: float  # 0..1


@dataclass(frozen=True)
class AtRiskInterval:
    t_start: float
    t_end: float
    n_at_risk: int


def reconstruct_ipd(
    *,
    km_points: list[KMPoint],
    at_risk: list[AtRiskInterval],
    arm: Literal["treatment", "control"],
    trial_id: str,
    profile: CovariateKey,
    rng_seed: int,
) -> list[Tier2Record]:
    """Reconstruct per-individual time-to-event records from a digitised KM curve.

    Simplified Guyot: distribute the interval's n_at_risk individuals across the
    KM-curve drops within the interval; events occur where surv decreases,
    censorings fill the rest. Returns one Tier2Record per individual.
    """
    if not km_points or not at_risk:
        return []
    rng = np.random.default_rng(rng_seed)
    records: list[Tier2Record] = []
    subj = 0

    sorted_pts = sorted(km_points, key=lambda p: p.time)
    for interval in at_risk:
        pts = [p for p in sorted_pts if interval.t_start <= p.time <= interval.t_end]
        if len(pts) < 2:
            continue
        s_start = pts[0].surv
        s_end = pts[-1].surv
        n = interval.n_at_risk
        n_events_expected = int(round(n * (s_start - s_end) / max(s_start, 1e-12)))
        n_censor = n - n_events_expected

        event_times = rng.uniform(pts[0].time, pts[-1].time, size=n_events_expected)
        censor_times = rng.uniform(pts[0].time, pts[-1].time, size=n_censor)

        for t in event_times:
            subj += 1
            records.append(Tier2Record(
                trial_id=trial_id, subject_id=f"{trial_id}-{arm}-{subj:06d}",
                arm=arm, time=float(t), event=1, covariates=profile,
                reconstructed=True,
            ))
        for t in censor_times:
            subj += 1
            records.append(Tier2Record(
                trial_id=trial_id, subject_id=f"{trial_id}-{arm}-{subj:06d}",
                arm=arm, time=float(t), event=0, covariates=profile,
                reconstructed=True,
            ))
    return records
