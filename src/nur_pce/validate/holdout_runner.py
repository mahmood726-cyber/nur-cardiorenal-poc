"""Held-out FINEARTS-HF validation orchestrator.

Wires the spec §9 validation contract end-to-end:
    1. Leakage gate: assert holdout NCT does not appear in training inputs.
    2. Fit PCE on training Tier-1 only.
    3. Compute pooled fixed-effect MA + meta-regression baselines on training.
    4. For each holdout subgroup, predict log-HR from each method.
    5. Score each method against the holdout truth (RMSE + 95% CrI calibration).
    6. Emit JSON report at <out_dir>/validation_report.json.

Reuses primitives from validate/baselines, validate/holdout, model/hte_bayes,
and model/diagnostics.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from nur_pce.ingest.subgroups import load_tier1
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs
from nur_pce.model.diagnostics import gate_diagnostics
from nur_pce.validate.baselines import pooled_fixed_effect, meta_regression
from nur_pce.validate.holdout import assert_no_leakage, score_predictions
from nur_pce.schema import Tier1Row


def _x_for_row(row: Tier1Row) -> np.ndarray:
    """Encode the same two-feature X-vector used by pipeline.py."""
    return np.array([
        1.0 if row.subgroup_key.t2dm else 0.0,
        1.0 if row.subgroup_key.sex == "M" else 0.0,
    ])


def _flatten_chains(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 2:
        return arr.reshape(-1)
    return arr.reshape(-1, *arr.shape[2:])


def run_holdout_validation(
    *,
    train_tier1_path: Path,
    holdout_tier1_path: Path,
    out_dir: Path,
) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    holdout = load_tier1(holdout_tier1_path)
    if not holdout:
        raise ValueError(f"No holdout rows in {holdout_tier1_path}")
    holdout_nct = holdout[0].trial_id

    assert_no_leakage(
        training_files=[Path(train_tier1_path)],
        holdout_nct=holdout_nct,
    )

    train = load_tier1(train_tier1_path)
    trial_to_idx: dict[str, int] = {}
    rows = []
    for r in train:
        if r.trial_id not in trial_to_idx:
            trial_to_idx[r.trial_id] = len(trial_to_idx) + 1
        rows.append({"y": r.log_hr, "s": r.se,
                     "trial": trial_to_idx[r.trial_id],
                     "X": _x_for_row(r).tolist()})
    inputs = HTEFitInputs(
        y=np.array([r["y"] for r in rows]),
        s=np.array([r["s"] for r in rows]),
        trial=np.array([r["trial"] for r in rows], dtype=int),
        n_trials=len(trial_to_idx),
        X=np.array([r["X"] for r in rows]),
    )

    fit = fit_hte(inputs, iter_warmup=2000, iter_sampling=1000, chains=4,
                  seed=1, target_accept=0.99)
    diag = fit.diagnostics()
    gate_diagnostics(diag)

    posterior = fit.idata.posterior
    mu_draws = _flatten_chains(posterior["mu"].values)
    beta_draws = _flatten_chains(posterior["beta"].values)

    truth = np.array([r.log_hr for r in holdout])
    pce_mean = np.empty(len(holdout))
    pce_lo = np.empty(len(holdout))
    pce_hi = np.empty(len(holdout))
    for i, r in enumerate(holdout):
        x = _x_for_row(r)
        log_hr_draws = mu_draws + beta_draws @ x
        pce_mean[i] = float(log_hr_draws.mean())
        pce_lo[i] = float(np.quantile(log_hr_draws, 0.025))
        pce_hi[i] = float(np.quantile(log_hr_draws, 0.975))

    pooled = pooled_fixed_effect(log_hrs=inputs.y, ses=inputs.s)
    pfe_mean = np.full(len(holdout), pooled["log_hr"])
    pfe_se = pooled["se"]
    pfe_lo = pfe_mean - 1.96 * pfe_se
    pfe_hi = pfe_mean + 1.96 * pfe_se

    mr_X = inputs.X
    mr = meta_regression(log_hrs=inputs.y, ses=inputs.s, X=mr_X)
    mr_intercept = mr["intercept"]
    mr_coefs = np.array(mr["coefs"])
    mr_se_int = mr["se_intercept"]
    mr_se_coefs = np.array(mr["se_coefs"])
    mr_mean = np.empty(len(holdout))
    mr_se = np.empty(len(holdout))
    for i, r in enumerate(holdout):
        x = _x_for_row(r)
        mr_mean[i] = float(mr_intercept + mr_coefs @ x)
        mr_se[i] = float(np.sqrt(mr_se_int**2 + (x**2 @ mr_se_coefs**2)))
    mr_lo = mr_mean - 1.96 * mr_se
    mr_hi = mr_mean + 1.96 * mr_se

    methods = {
        "pce": score_predictions(
            truth_log_hr=truth, pred_log_hr_mean=pce_mean,
            pred_log_hr_lo=pce_lo, pred_log_hr_hi=pce_hi,
        ),
        "pooled_fe": score_predictions(
            truth_log_hr=truth, pred_log_hr_mean=pfe_mean,
            pred_log_hr_lo=pfe_lo, pred_log_hr_hi=pfe_hi,
        ),
        "meta_regression": score_predictions(
            truth_log_hr=truth, pred_log_hr_mean=mr_mean,
            pred_log_hr_lo=mr_lo, pred_log_hr_hi=mr_hi,
        ),
    }

    report = {
        "schema_version": "0.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_trial_id": holdout_nct,
        "n_holdout_subgroups": len(holdout),
        "n_training_rows": len(train),
        "n_training_trials": len(trial_to_idx),
        "mcmc_diagnostics": diag,
        "methods": methods,
        "per_subgroup": [
            {
                "subgroup_key": r.subgroup_key.model_dump(),
                "truth_log_hr": float(truth[i]),
                "pce_mean": float(pce_mean[i]),
                "pce_95_cri": [float(pce_lo[i]), float(pce_hi[i])],
                "pooled_fe_mean": float(pfe_mean[i]),
                "meta_reg_mean": float(mr_mean[i]),
            }
            for i, r in enumerate(holdout)
        ],
    }
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return out_dir
