from __future__ import annotations
import json
from pathlib import Path
import pytest
from nur_pce.ingest.subgroups import load_tier1, Tier1ValidationError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tier1_synth.json"


def test_load_tier1_validates_synth():
    rows = load_tier1(FIXTURE)
    assert len(rows) == 2
    assert all(r.extractor and r.verifier for r in rows)


def test_load_tier1_rejects_missing_verifier(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "outcome": "composite_cardiorenal",
        "rows": [{
            "trial_id": "NCT02540993",
            "subgroup_key": {
                "age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
                "t2dm": True, "uacr_band": "300-1000", "nyha": "I-II",
                "region": "USA"
            },
            "log_hr": -0.30, "se": 0.10,
            "outcome": "composite_cardiorenal",
            "extractor": "x", "source_doc": "y"
        }]
    }))
    with pytest.raises(Tier1ValidationError):
        load_tier1(bad)


def test_load_tier1_rejects_negative_se(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "outcome": "composite_cardiorenal",
        "rows": [{
            "trial_id": "NCT02540993",
            "subgroup_key": {
                "age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
                "t2dm": True, "uacr_band": "300-1000", "nyha": "I-II",
                "region": "USA"
            },
            "log_hr": -0.30, "se": -0.10,
            "outcome": "composite_cardiorenal",
            "extractor": "x", "verifier": "y", "source_doc": "z"
        }]
    }))
    with pytest.raises(Tier1ValidationError):
        load_tier1(bad)
