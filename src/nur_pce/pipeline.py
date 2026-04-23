"""End-to-end pipeline orchestrator.

CLI subcommands:
    run-synth         — pipeline on synthetic fixtures (used by tests + CI)
    run-validation    — held-out FINEARTS-HF validation (v0.2 spec §3a item 4);
                        emits per-method RMSE + 95% CrI calibration vs truth

Posterior flow (post-review fix `c13d3cf+`):
    fit_hte (PyMC) -> idata.posterior -> per-cell log_hr_draws via
    log_hr = mu + X_cell . (beta + gamma).  Cube cells now reflect actual
    posterior, not fabricated draws.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from nur_pce.ingest.subgroups import load_tier1
from nur_pce.transport.populations import load_populations
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs
from nur_pce.model.diagnostics import gate_diagnostics
from nur_pce.output.cube import build_cell, write_cube
from nur_pce.schema import (
    CovariateKey, AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES,
)


def _x_for_cell(t2dm: bool, sex: str) -> np.ndarray:
    """Encode the synth pipeline's two-feature X-vector for a given cell.

    The synth fixture uses two binary features: T2DM and male-sex. The fit's
    X matrix uses the same encoding, so the cell's X must match.
    """
    return np.array([1.0 if t2dm else 0.0, 1.0 if sex == "M" else 0.0])


def _flatten_chains(arr: np.ndarray) -> np.ndarray:
    """Flatten posterior shape (chain, draw, ...) to (chain*draw, ...)."""
    if arr.ndim <= 2:
        return arr.reshape(-1)
    return arr.reshape(-1, *arr.shape[2:])


def run_synth_pipeline(*, out_dir: Path, fixtures_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tier1 = load_tier1(fixtures_dir / "tier1_synth.json")
    populations = load_populations(fixtures_dir / "populations_synth.json")

    trial_to_idx: dict[str, int] = {}
    rows: list[dict] = []
    for r in tier1:
        if r.trial_id not in trial_to_idx:
            trial_to_idx[r.trial_id] = len(trial_to_idx) + 1
        rows.append({
            "y": r.log_hr, "s": r.se, "trial": trial_to_idx[r.trial_id],
            "X": _x_for_cell(r.subgroup_key.t2dm,
                             r.subgroup_key.sex).tolist(),
        })
    inputs = HTEFitInputs(
        y=np.array([r["y"] for r in rows]),
        s=np.array([r["s"] for r in rows]),
        trial=np.array([r["trial"] for r in rows], dtype=int),
        n_trials=len(trial_to_idx),
        X=np.array([r["X"] for r in rows]),
    )
    fit = fit_hte(inputs, iter_warmup=1500, iter_sampling=1000, chains=4,
                  seed=1, target_accept=0.95)
    diag = fit.diagnostics()
    gate_diagnostics(diag)

    posterior = fit.idata.posterior
    mu_draws = _flatten_chains(posterior["mu"].values)            # (S,)
    beta_draws = _flatten_chains(posterior["beta"].values)        # (S, P)
    gamma_draws = _flatten_chains(posterior["gamma"].values)      # (S, P)
    coef_draws = beta_draws + gamma_draws                          # (S, P)
    var_sampling = float(np.mean(inputs.s ** 2))

    cells = []
    region = next(iter(populations))
    for age in AGE_BANDS[:1]:
        for sex in ("M", "F"):
            for egfr in EGFR_BANDS[:1]:
                for t2dm in (True, False):
                    for uacr in UACR_BANDS[:1]:
                        for nyha in NYHA_CLASSES[:1]:
                            key = CovariateKey(
                                age_band=age, sex=sex, eGFR_band=egfr,
                                t2dm=t2dm, uacr_band=uacr, nyha=nyha,
                                region=region,
                            )
                            x_cell = _x_for_cell(t2dm, sex)
                            log_hr_draws = mu_draws + coef_draws @ x_cell
                            var_hte = float(np.var(coef_draws @ x_cell))
                            cells.append(build_cell(
                                key=key, log_hr_draws=log_hr_draws, tier=1,
                                var_sampling=var_sampling, var_hte=var_hte,
                                var_transport=0.0,
                            ))
    write_cube(
        path=out_dir / "posterior_cube.json", cells=cells,
        diagnostics=diag, drug="finerenone", comparator="placebo",
        outcome="composite_cardiorenal",
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(prog="nur_pce")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_synth = sub.add_parser("run-synth")
    run_synth.add_argument("--out", type=Path, default=Path("outputs"))
    run_synth.add_argument("--fixtures", type=Path, default=Path("fixtures"))

    run_val = sub.add_parser("run-validation")
    run_val.add_argument("--train", type=Path,
                         default=Path("fixtures/tier1_train_synth.json"))
    run_val.add_argument("--holdout", type=Path,
                         default=Path("fixtures/tier1_holdout_finearts_synth.json"))
    run_val.add_argument("--out", type=Path,
                         default=Path("outputs/validation"))

    args = parser.parse_args()
    if args.cmd == "run-synth":
        out = run_synth_pipeline(out_dir=args.out, fixtures_dir=args.fixtures)
        print(f"Cube written to {out / 'posterior_cube.json'}")
    elif args.cmd == "run-validation":
        from nur_pce.validate.holdout_runner import run_holdout_validation
        out = run_holdout_validation(
            train_tier1_path=args.train,
            holdout_tier1_path=args.holdout,
            out_dir=args.out,
        )
        print(f"Validation report written to {out / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
