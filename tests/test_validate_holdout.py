from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from nur_pce.validate.holdout import (
    assert_no_leakage, score_predictions, LeakageDetected,
)


def test_assert_no_leakage_passes_when_holdout_absent(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps({"trial_id": "NCT02540993"}))
    assert_no_leakage(training_files=[a], holdout_nct="NCT04435626")


def test_assert_no_leakage_raises_when_holdout_present(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps({"trial_id": "NCT04435626"}))
    with pytest.raises(LeakageDetected):
        assert_no_leakage(training_files=[a], holdout_nct="NCT04435626")


def test_score_predictions_computes_rmse_and_calibration():
    truth = np.array([-0.30, -0.20, -0.40])
    pred_mean = np.array([-0.28, -0.18, -0.45])
    pred_lo = np.array([-0.45, -0.30, -0.60])
    pred_hi = np.array([-0.15, -0.10, -0.30])
    res = score_predictions(
        truth_log_hr=truth, pred_log_hr_mean=pred_mean,
        pred_log_hr_lo=pred_lo, pred_log_hr_hi=pred_hi,
    )
    assert res["rmse"] == pytest.approx(
        float(np.sqrt(((pred_mean - truth) ** 2).mean())), abs=1e-9
    )
    assert 0.0 <= res["calibration"] <= 1.0
    assert res["calibration"] == pytest.approx(1.0)
