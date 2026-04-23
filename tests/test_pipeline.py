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
    # Posterior must flow through to per-cell HRs (regression guard for the
    # pre-fix bug where all cells were filled from a single fabricated draw).
    hr_means = sorted({round(c["hr_mean"], 4) for c in cube["cells"]})
    assert len(hr_means) > 1, (
        f"all {len(cube['cells'])} cells share the same HR — posterior "
        f"likely not wired through pipeline"
    )
    # Uncertainty decomposition must report the actual sampling variance,
    # not the placeholder 0.01 from the pre-fix pipeline.
    sampling_vars = {round(c["uncertainty_decomp"]["sampling"], 6)
                     for c in cube["cells"]}
    assert sampling_vars != {0.01}, (
        "var_sampling is the placeholder 0.01; expected mean(s^2) from inputs"
    )
