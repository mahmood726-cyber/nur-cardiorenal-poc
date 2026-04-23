"""Load target-population covariate marginals (synthetic in v0.1; IHME/WHO/WB
ingest for v0.2)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

NORMALISE_TOL = 0.05  # accept marginals within 5% of summing to 1; renormalise


class PopulationLoadError(ValueError):
    pass


@dataclass(frozen=True)
class PopulationMarginals:
    region: str
    covariate_marginals: dict[str, dict[str, float]]
    source: str


def load_populations(path: Path) -> dict[str, PopulationMarginals]:
    payload = json.loads(Path(path).read_text())
    out: dict[str, PopulationMarginals] = {}
    for region, info in payload.get("populations", {}).items():
        marginals: dict[str, dict[str, float]] = {}
        for cov, dist in info["covariate_marginals"].items():
            total = sum(dist.values())
            if abs(total - 1.0) > NORMALISE_TOL:
                raise PopulationLoadError(
                    f"{region}/{cov} marginal sums to {total:.4f}, "
                    f"outside tolerance {NORMALISE_TOL}"
                )
            marginals[cov] = {k: v / total for k, v in dist.items()}
        out[region] = PopulationMarginals(
            region=region,
            covariate_marginals=marginals,
            source=info.get("source", "unknown"),
        )
    return out
