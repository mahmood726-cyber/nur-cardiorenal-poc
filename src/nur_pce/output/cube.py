"""Build CubeCells from posterior draws and write the JSON cube."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from nur_pce.schema import (
    CovariateKey, CubeCell, Cube, UncertaintyDecomp,
    AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES, REGIONS,
)


def build_cell(
    *, key: CovariateKey, log_hr_draws: np.ndarray, tier: int,
    var_sampling: float, var_hte: float, var_transport: float,
) -> CubeCell:
    hr_draws = np.exp(log_hr_draws)
    return CubeCell(
        key=key,
        hr_mean=float(hr_draws.mean()),
        hr_credible_95=(
            float(np.quantile(hr_draws, 0.025)),
            float(np.quantile(hr_draws, 0.975)),
        ),
        p_hr_lt_1=float((hr_draws < 1.0).mean()),
        tier=tier,  # type: ignore[arg-type]
        uncertainty_decomp=UncertaintyDecomp(
            sampling=var_sampling, hte=var_hte, transport=var_transport,
        ),
    )


def write_cube(
    *, path: Path, cells: list[CubeCell], diagnostics: dict[str, float],
    drug: str, comparator: str, outcome: str,
) -> None:
    cube = Cube(
        schema_version="0.1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        drug=drug, comparator=comparator, outcome=outcome,
        covariates=["age_band", "sex", "eGFR_band", "t2dm",
                    "uacr_band", "nyha", "region"],
        cells=cells,
        diagnostics=diagnostics,
    )
    Path(path).write_text(cube.model_dump_json(indent=2))
