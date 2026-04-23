from __future__ import annotations
import json
from pathlib import Path
import pytest
from nur_pce.validate.holdout_runner import run_holdout_validation


pytestmark = pytest.mark.slow


def test_holdout_validation_emits_three_method_report(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    out = run_holdout_validation(
        train_tier1_path=fixtures / "tier1_train_synth.json",
        holdout_tier1_path=fixtures / "tier1_holdout_finearts_synth.json",
        out_dir=tmp_path,
    )
    report_path = out / "validation_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert report["holdout_trial_id"] == "NCT04435626"
    assert report["n_holdout_subgroups"] == 5
    assert set(report["methods"].keys()) == {"pce", "pooled_fe", "meta_regression"}
    for name, m in report["methods"].items():
        assert "rmse" in m, f"{name} missing rmse"
        assert "calibration" in m, f"{name} missing calibration"
        assert m["n"] == 5, f"{name} wrong n"
        assert m["rmse"] >= 0
        assert 0.0 <= m["calibration"] <= 1.0

    diag = report["mcmc_diagnostics"]
    assert diag["rhat_max"] <= 1.05
    assert diag["ess_min"] >= 200


def test_holdout_validation_fails_closed_on_leakage(tmp_path):
    """If FINEARTS NCT appears in training inputs, raise LeakageDetected."""
    from nur_pce.validate.holdout import LeakageDetected
    fixtures = Path(__file__).parent.parent / "fixtures"
    bad_train = tmp_path / "leaky_train.json"
    payload = json.loads(
        (fixtures / "tier1_train_synth.json").read_text()
    )
    payload["rows"].append({
        "trial_id": "NCT04435626",
        "subgroup_key": {"age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
                         "t2dm": True, "uacr_band": "300-1000",
                         "nyha": "I-II", "region": "USA"},
        "log_hr": -0.30, "se": 0.10,
        "outcome": "composite_cardiorenal",
        "extractor": "leak", "verifier": "leak", "source_doc": "leak",
    })
    bad_train.write_text(json.dumps(payload))
    with pytest.raises(LeakageDetected):
        run_holdout_validation(
            train_tier1_path=bad_train,
            holdout_tier1_path=fixtures / "tier1_holdout_finearts_synth.json",
            out_dir=tmp_path,
        )
