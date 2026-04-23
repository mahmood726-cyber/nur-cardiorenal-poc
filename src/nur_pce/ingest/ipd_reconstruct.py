"""Tier-2 IPD reconstruction via the Guyot et al. 2012 algorithm.

Reconstructs n = at_risk[0].n_at_risk individuals — the original enrolled cohort.
Each individual appears exactly once with their single event-or-censor time.

For each interval i:
  n_lost_in_interval = at_risk[i].n_at_risk - at_risk[i+1].n_at_risk
                       (or at_risk[i].n_at_risk if i is the last interval —
                        all remaining are administratively censored)
  Of those lost, the KM curve drop tells us how many were events vs censorings:
  n_events_in_interval = round(n_at_start * (s_start - s_end) / s_start),
                         clipped to <= n_lost.

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

    Returns one Tier2Record per enrolled individual (n = at_risk[0].n_at_risk).
    """
    if not km_points or not at_risk:
        return []
    rng = np.random.default_rng(rng_seed)
    records: list[Tier2Record] = []
    subj = 0

    sorted_pts = sorted(km_points, key=lambda p: p.time)
    for i, interval in enumerate(at_risk):
        pts = [p for p in sorted_pts
               if interval.t_start <= p.time <= interval.t_end]
        n_at_start = interval.n_at_risk
        # Patients who leave the risk set during this interval:
        if i + 1 < len(at_risk):
            n_at_end = at_risk[i + 1].n_at_risk
        else:
            n_at_end = 0  # last interval — all remaining censored at t_end
        n_lost = max(n_at_start - n_at_end, 0)
        if n_lost == 0 or len(pts) < 2:
            # No KM info or no losses — emit n_lost censorings if any, at t_end
            for _ in range(n_lost):
                subj += 1
                records.append(Tier2Record(
                    trial_id=trial_id,
                    subject_id=f"{trial_id}-{arm}-{subj:06d}",
                    arm=arm, time=float(interval.t_end), event=0,
                    covariates=profile, reconstructed=True,
                ))
            continue
        s_start = pts[0].surv
        s_end = pts[-1].surv
        # Events implied by KM drop, clipped to <= n_lost:
        n_events = int(round(
            n_at_start * (s_start - s_end) / max(s_start, 1e-12)
        ))
        n_events = max(0, min(n_events, n_lost))
        n_censor = n_lost - n_events

        event_times = rng.uniform(pts[0].time, pts[-1].time, size=n_events)
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
