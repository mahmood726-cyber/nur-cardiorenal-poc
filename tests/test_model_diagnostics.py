from __future__ import annotations
import pytest
from nur_pce.model.diagnostics import gate_diagnostics, DiagnosticsFailure


def test_pass_within_thresholds():
    gate_diagnostics({"rhat_max": 1.005, "ess_min": 800, "divergent": 0})


def test_fail_on_rhat_above_threshold():
    with pytest.raises(DiagnosticsFailure, match="R_hat"):
        gate_diagnostics({"rhat_max": 1.05, "ess_min": 800, "divergent": 0})


def test_fail_on_ess_below_threshold():
    with pytest.raises(DiagnosticsFailure, match="ESS"):
        gate_diagnostics({"rhat_max": 1.005, "ess_min": 50, "divergent": 0})


def test_fail_on_divergent_transitions():
    with pytest.raises(DiagnosticsFailure, match="divergent"):
        gate_diagnostics({"rhat_max": 1.005, "ess_min": 800, "divergent": 5})
