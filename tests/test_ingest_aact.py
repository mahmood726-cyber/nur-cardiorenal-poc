from __future__ import annotations
from pathlib import Path
import pytest
from nur_pce.ingest.aact import find_drug_trials, AACTSnapshotMissing

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "aact_synth"


def test_find_finerenone_trials_returns_three_rcts():
    trials = find_drug_trials(FIXTURE_ROOT, drug="finerenone")
    nct_ids = sorted(t.nct_id for t in trials)
    assert nct_ids == ["NCT02540993", "NCT02545049", "NCT04435626"]
    for t in trials:
        assert t.drug.lower() == "finerenone"
        assert t.comparator.lower() == "placebo"


def test_find_drug_trials_excludes_unrelated():
    trials = find_drug_trials(FIXTURE_ROOT, drug="finerenone")
    assert "NCT99999999" not in {t.nct_id for t in trials}


def test_find_drug_trials_fails_closed_on_missing_snapshot(tmp_path):
    with pytest.raises(AACTSnapshotMissing):
        find_drug_trials(tmp_path / "does_not_exist", drug="finerenone")
