"""Held-out validation: leakage gate + scoring metrics.

Per spec §11: the leakage gate hashes training-input file paths and asserts
no held-out NCT appears in any training input. Failure halts the pipeline.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


class LeakageDetected(RuntimeError):
    pass


def assert_no_leakage(*, training_files: list[Path], holdout_nct: str) -> None:
    for f in training_files:
        text = Path(f).read_text(errors="ignore")
        if holdout_nct in text:
            raise LeakageDetected(
                f"Held-out NCT {holdout_nct} found in training input {f}. "
                f"Validation invalid."
            )


def score_predictions(
    *, truth_log_hr: np.ndarray, pred_log_hr_mean: np.ndarray,
    pred_log_hr_lo: np.ndarray, pred_log_hr_hi: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth_log_hr)
    mean = np.asarray(pred_log_hr_mean)
    lo = np.asarray(pred_log_hr_lo)
    hi = np.asarray(pred_log_hr_hi)
    rmse = float(np.sqrt(((mean - truth) ** 2).mean()))
    inside = ((truth >= lo) & (truth <= hi)).mean()
    return {
        "rmse": rmse,
        "calibration": float(inside),
        "n": int(truth.size),
    }
