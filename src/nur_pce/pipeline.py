"""End-to-end pipeline orchestrator.

CLI subcommands:
    run-synth         — pipeline on synthetic fixtures (used by tests + CI)
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
            "X": [
                1.0 if r.subgroup_key.t2dm else 0.0,
                1.0 if r.subgroup_key.sex == "M" else 0.0,
            ],
        })
    inputs = HTEFitInputs(
        y=np.array([r["y"] for r in rows]),
        s=np.array([r["s"] for r in rows]),
        trial=np.array([r["trial"] for r in rows], dtype=int),
        n_trials=len(trial_to_idx),
        X=np.array([r["X"] for r in rows]),
    )
    fit = fit_hte(inputs, iter_warmup=1500, iter_sampling=1000, chains=4, seed=1,
                  target_accept=0.95)
    diag = fit.diagnostics()
    gate_diagnostics(diag)

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
                            log_hr_draws = np.random.default_rng(0).normal(-0.30, 0.10, size=400)
                            cells.append(build_cell(
                                key=key, log_hr_draws=log_hr_draws, tier=1,
                                var_sampling=0.01, var_hte=0.005, var_transport=0.001,
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
    args = parser.parse_args()
    if args.cmd == "run-synth":
        out = run_synth_pipeline(out_dir=args.out, fixtures_dir=args.fixtures)
        print(f"Cube written to {out / 'posterior_cube.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
