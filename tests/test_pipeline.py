from __future__ import annotations
import json
from pathlib import Path
import pytest
from nur_pce.pipeline import run_synth_pipeline


pytestmark = pytest.mark.slow


def test_synth_pipeline_produces_valid_cube(tmp_path):
    out = run_synth_pipeline(
        out_dir=tmp_path,
        fixtures_dir=Path(__file__).parent.parent / "fixtures",
    )
    cube_path = out / "posterior_cube.json"
    assert cube_path.exists()
    cube = json.loads(cube_path.read_text())
    assert cube["schema_version"] == "0.1"
    assert len(cube["cells"]) > 0
    for cell in cube["cells"]:
        assert cell["hr_mean"] > 0
        assert cell["tier"] in (1, 2, 3)
