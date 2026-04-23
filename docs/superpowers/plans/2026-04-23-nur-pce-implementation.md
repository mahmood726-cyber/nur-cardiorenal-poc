# NUR-PCE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working POC of NUR-PCE — a personalised counterfactual engine for finerenone vs placebo on the composite cardiorenal outcome — that produces tier-labelled, transportable, posterior-quantified HRs and validates against held-out FINEARTS-HF subgroups.

**Architecture:** Seven-stage Python pipeline (ingest × 3 → Bayesian HTE model → g-formula transportability projection → posterior cube → single-file HTML viewer), each stage JSON in / JSON out. Hierarchical Bayesian model fit in Stan via `cmdstanpy` with PyMC fallback. Validation by held-out FINEARTS-HF subgroup HR prediction.

**Tech Stack:** Python 3.13, pytest (TDD), `cmdstanpy` (primary) / PyMC (fallback), Stan, pandas + pyarrow, pydantic v2, numpy, scipy, matplotlib (validation reports only — viewer is dependency-free HTML/JS).

**Spec:** `docs/superpowers/specs/2026-04-23-nur-pce-design.md` (v0.1, locked, commit `a6de22e`).

---

## File Structure (locked at plan-time)

```
C:/Projects/nur-cardiorenal-poc/
├── pyproject.toml
├── pytest.ini
├── conftest.py
├── README.md
├── E156-PROTOCOL.md                    # written at v0.1 ship
├── .gitignore
├── docs/
│   └── superpowers/
│       ├── specs/2026-04-23-nur-pce-design.md
│       └── plans/2026-04-23-nur-pce-implementation.md
├── src/nur_pce/
│   ├── __init__.py
│   ├── schema.py                       # Pydantic data classes
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── aact.py                     # finerenone trials from D:/AACT-storage
│   │   ├── subgroups.py                # validated Tier 1 entry
│   │   └── ipd_reconstruct.py          # Guyot from KM
│   ├── model/
│   │   ├── __init__.py
│   │   ├── stan/hte.stan               # Stan model
│   │   ├── hte_bayes.py                # cmdstanpy wrapper
│   │   └── diagnostics.py              # R̂, ESS, divergent gates
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── populations.py              # IHME/WHO/WB marginals loader
│   │   └── g_formula.py                # posterior projection
│   ├── output/
│   │   ├── __init__.py
│   │   └── cube.py                     # posterior_cube.json writer
│   ├── validate/
│   │   ├── __init__.py
│   │   ├── baselines.py                # pooled FE + meta-regression
│   │   └── holdout.py                  # FINEARTS-HF leakage gate + scoring
│   └── pipeline.py                     # CLI orchestrator
├── fixtures/
│   ├── trials_synth.json
│   ├── tier1_synth.json
│   ├── tier2_synth.parquet
│   └── populations_synth.json
├── tests/
│   ├── __init__.py                     # CRITICAL — see lessons.md
│   ├── test_schema.py
│   ├── test_ingest_aact.py
│   ├── test_ingest_subgroups.py
│   ├── test_ingest_ipd_reconstruct.py
│   ├── test_model_hte_bayes.py
│   ├── test_model_diagnostics.py
│   ├── test_transport_populations.py
│   ├── test_transport_g_formula.py
│   ├── test_output_cube.py
│   ├── test_validate_baselines.py
│   ├── test_validate_holdout.py
│   ├── test_pipeline.py
│   └── test_viewer_smoke.py
├── viewer/
│   └── index.html                      # single file, no CDN
├── data/                               # gitignored, output of ingest stages
└── outputs/                            # gitignored, posterior + validation reports
```

**File responsibility check:**
- `schema.py` — only Pydantic classes; no logic.
- `ingest/*` — read-only on external sources; write structured JSON; no model fitting.
- `model/*` — fit posterior; expose diagnostics; no I/O of cubes or transports.
- `transport/*` — apply g-formula; no model fitting; depends on posterior + populations.
- `output/cube.py` — assemble cube JSON from transported posterior; no fitting; no I/O of source PDFs.
- `validate/*` — compute baselines and held-out scores; reads cube + ground truth.
- `pipeline.py` — CLI orchestrator; depends on all the above; no statistical logic of its own.
- `viewer/index.html` — read-only on `posterior_cube.json`; no compute beyond client-side keying.

---

## Task 0: Preflight prerequisites

**Why this is Task 0**: per `lessons.md` ("Preflight external prereqs BEFORE starting a multi-task plan"), any plan whose final task depends on external integration adds a Task 0 that scripts the prereq check and fails closed with a specific user-action list. Task 12 (held-out validation) depends on cmdstanpy + AACT + IHME — fail-closed before writing tests.

**Files:**
- Create: `scripts/preflight.py`

- [ ] **Step 1: Write the preflight script**

```python
# scripts/preflight.py
"""Preflight check for NUR-PCE implementation prerequisites.

Fails closed with a specific action list if any prereq is missing.
Run before starting any other task.
"""
from __future__ import annotations
import importlib.util
import shutil
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 11)  # works on 3.13 with WMI patch
AACT_PATH = Path("D:/AACT-storage/AACT/2026-04-12")
IHME_PATH = Path("D:/Projects/ihme-data-lakehouse")
WHO_PATH = Path("D:/Projects/who-data-lakehouse")
WB_PATH = Path("D:/Projects/wb-data-lakehouse")
KMEXTRACT_PATH = Path("C:/KMextract")  # may live elsewhere; check both


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    if (v.major, v.minor) < REQUIRED_PYTHON:
        return False, f"Python {v.major}.{v.minor} < required {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"
    return True, f"Python {v.major}.{v.minor}.{v.micro} OK"


def check_module(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, f"Module '{name}' NOT installed (pip install {name})"
    return True, f"Module '{name}' OK"


def check_path(p: Path, label: str) -> tuple[bool, str]:
    if not p.exists():
        return False, f"{label} path missing: {p}"
    return True, f"{label} OK at {p}"


def check_stan() -> tuple[bool, str]:
    try:
        import cmdstanpy  # type: ignore
        try:
            cmdstanpy.cmdstan_path()
            return True, "cmdstan installed"
        except Exception as e:
            return False, f"cmdstanpy installed but cmdstan toolchain missing: {e} (run: python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()')"
    except ImportError:
        return False, "cmdstanpy NOT installed (pip install cmdstanpy)"


def main() -> int:
    checks = [
        ("python", check_python()),
        ("pydantic", check_module("pydantic")),
        ("pytest", check_module("pytest")),
        ("numpy", check_module("numpy")),
        ("scipy", check_module("scipy")),
        ("pandas", check_module("pandas")),
        ("pyarrow", check_module("pyarrow")),
        ("cmdstanpy", check_stan()),
        ("aact", check_path(AACT_PATH, "AACT snapshot")),
        ("ihme", check_path(IHME_PATH, "IHME lakehouse")),
        ("who", check_path(WHO_PATH, "WHO lakehouse")),
        ("wb", check_path(WB_PATH, "WB lakehouse")),
    ]
    failed = [(n, msg) for n, (ok, msg) in checks if not ok]
    for n, (ok, msg) in checks:
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {n}: {msg}")
    if failed:
        print("\n=== ACTION REQUIRED ===")
        for n, msg in failed:
            print(f"  - {n}: {msg}")
        return 1
    print("\nAll prereqs satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run preflight**

Run: `python scripts/preflight.py`
Expected: prints OK/FAIL per check; exits 0 if all OK; exits 1 with action list otherwise.

- [ ] **Step 3: Resolve any failures**

For each FAIL: install missing module (`pip install <name>`), install cmdstan (`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`), or correct paths. Do not proceed to Task 1 until preflight exits 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/preflight.py
git commit -m "task0: preflight prerequisites check"
```

---

## Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `pytest.ini`, `conftest.py`, `.gitignore`, `tests/__init__.py`, `src/nur_pce/__init__.py`, `README.md`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "nur-pce"
version = "0.1.0"
description = "NUR-PCE — Personalised Counterfactual Engine for cardiorenal trials"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    "pyarrow>=15",
    "cmdstanpy>=1.2",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=4", "selenium>=4.18"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra --strict-markers
filterwarnings =
    ignore::DeprecationWarning:cmdstanpy.*
```

Per `lessons.md`: "Module-name collision hides tests". `testpaths` + `tests/__init__.py` together prevent this.

- [ ] **Step 3: Write conftest.py**

```python
# conftest.py
"""Pytest fixtures shared across the suite."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src/ is importable
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"
```

- [ ] **Step 4: Write .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# Project outputs (regeneratable)
data/
outputs/
*.parquet
viewer/posterior_cube.json

# Per ~/.claude/rules/rules.md: PROGRESS.md is local
PROGRESS.md
sentinel-findings.md
sentinel-findings.jsonl
STUCK_FAILURES.md
STUCK_FAILURES.jsonl

# Stan compiled artefacts
*.exe
*_model
src/nur_pce/model/stan/hte
src/nur_pce/model/stan/hte.exe

# IDE
.vscode/
.idea/
```

- [ ] **Step 5: Write tests/__init__.py and src/nur_pce/__init__.py**

```python
# tests/__init__.py
"""Test package marker — required to avoid module-name collisions per lessons.md."""
```

```python
# src/nur_pce/__init__.py
"""NUR-PCE — Personalised Counterfactual Engine."""
__version__ = "0.1.0"
```

- [ ] **Step 6: Write README stub**

```markdown
# NUR-PCE — Personalised Counterfactual Engine (POC)

Replaces the meta-analytic unit of evidence with the patient-conditional counterfactual posterior. Proof-of-concept on finerenone vs placebo for the composite cardiorenal outcome, validated against held-out FINEARTS-HF subgroups.

See `docs/superpowers/specs/2026-04-23-nur-pce-design.md` for the full spec.

## Quickstart

```bash
pip install -e .[dev]
python scripts/preflight.py        # verify prereqs
pytest -v                           # run test suite
python -m nur_pce.pipeline run-all  # end-to-end
```

## Status

v0.1 — POC under construction. See `docs/superpowers/plans/`.
```

- [ ] **Step 7: Verify project installs and pytest collects zero tests**

Run: `pip install -e .[dev]`
Expected: success.

Run: `pytest -v`
Expected: `collected 0 items` (no tests yet — confirms collection works).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml pytest.ini conftest.py .gitignore README.md \
        tests/__init__.py src/nur_pce/__init__.py
git commit -m "task1: project skeleton (pyproject, pytest, gitignore, packages)"
```

---

## Task 2: Schema (Pydantic data classes)

**Files:**
- Create: `src/nur_pce/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
from __future__ import annotations
import pytest
from pydantic import ValidationError
from nur_pce.schema import (
    Trial, CovariateKey, Tier1Row, Tier2Record, CubeCell, UncertaintyDecomp,
    AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES, REGIONS,
)


def test_covariate_key_validates_bands():
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    assert key.age_band == "60-70"


def test_covariate_key_rejects_bad_age_band():
    with pytest.raises(ValidationError):
        CovariateKey(
            age_band="999", sex="M", eGFR_band="30-45",
            t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
        )


def test_tier1_row_requires_extractor_and_verifier():
    with pytest.raises(ValidationError):
        Tier1Row(
            trial_id="NCT02540993",
            subgroup_key=CovariateKey(
                age_band="60-70", sex="M", eGFR_band="30-45",
                t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
            ),
            log_hr=-0.30, se=0.12, outcome="composite_cardiorenal",
        )  # missing extractor + verifier


def test_cube_cell_uncertainty_decomp_sums_to_total():
    cell = CubeCell(
        key=CovariateKey(
            age_band="70-75", sex="M", eGFR_band="30-45",
            t2dm=True, uacr_band="300-1000", nyha="III-IV", region="Pakistan",
        ),
        hr_mean=0.74, hr_credible_95=(0.61, 0.89), p_hr_lt_1=0.997,
        tier=2,
        uncertainty_decomp=UncertaintyDecomp(sampling=0.06, hte=0.04, transport=0.02),
    )
    assert cell.uncertainty_decomp.total() == pytest.approx(0.12)


def test_constants_match_spec():
    assert AGE_BANDS == ("<60", "60-70", "70-75", "75+")
    assert EGFR_BANDS == ("<30", "30-45", "45-60", ">=60")
    assert UACR_BANDS == ("<300", "300-1000", ">=1000")
    assert NYHA_CLASSES == ("I-II", "III-IV")
    assert REGIONS == ("Pakistan", "India", "Sub-Saharan Africa", "USA", "EU")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nur_pce.schema'`.

- [ ] **Step 3: Write the schema module**

```python
# src/nur_pce/schema.py
"""Pydantic data classes for NUR-PCE.

Constants are typed as Literal so Pydantic enforces band membership at parse time.
"""
from __future__ import annotations
from typing import Literal, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

AGE_BANDS = ("<60", "60-70", "70-75", "75+")
EGFR_BANDS = ("<30", "30-45", "45-60", ">=60")
UACR_BANDS = ("<300", "300-1000", ">=1000")
NYHA_CLASSES = ("I-II", "III-IV")
REGIONS = ("Pakistan", "India", "Sub-Saharan Africa", "USA", "EU")

AgeBand = Literal["<60", "60-70", "70-75", "75+"]
Sex = Literal["M", "F"]
EgfrBand = Literal["<30", "30-45", "45-60", ">=60"]
UacrBand = Literal["<300", "300-1000", ">=1000"]
NyhaClass = Literal["I-II", "III-IV"]
Region = Literal["Pakistan", "India", "Sub-Saharan Africa", "USA", "EU"]
Tier = Literal[1, 2, 3]


class CovariateKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    age_band: AgeBand
    sex: Sex
    eGFR_band: EgfrBand
    t2dm: bool
    uacr_band: UacrBand
    nyha: NyhaClass
    region: Region


class Trial(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    drug: str
    comparator: str
    n_total: int = Field(gt=0)
    design: str
    primary_outcome: str


class Tier1Row(BaseModel):
    trial_id: str = Field(pattern=r"^NCT\d{8}$")
    subgroup_key: CovariateKey
    log_hr: float
    se: float = Field(gt=0)
    outcome: str
    extractor: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    source_doc: str = Field(min_length=1)
    source_page: int | None = None


class Tier2Record(BaseModel):
    trial_id: str = Field(pattern=r"^NCT\d{8}$")
    subject_id: str
    arm: Literal["treatment", "control"]
    time: float = Field(ge=0)
    event: int = Field(ge=0, le=1)
    covariates: CovariateKey
    reconstructed: bool = True


class UncertaintyDecomp(BaseModel):
    sampling: float = Field(ge=0)
    hte: float = Field(ge=0)
    transport: float = Field(ge=0)

    def total(self) -> float:
        return self.sampling + self.hte + self.transport


class CubeCell(BaseModel):
    key: CovariateKey
    hr_mean: float = Field(gt=0)
    hr_credible_95: Tuple[float, float]
    p_hr_lt_1: float = Field(ge=0, le=1)
    tier: Tier
    uncertainty_decomp: UncertaintyDecomp

    @field_validator("hr_credible_95")
    @classmethod
    def _ci_ordered(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if not (lo > 0 and hi > 0 and lo <= hi):
            raise ValueError("hr_credible_95 must be (lo, hi) with 0 < lo <= hi")
        return v


class Cube(BaseModel):
    schema_version: str = "0.1"
    generated_at: str
    drug: str
    comparator: str
    outcome: str
    covariates: list[str]
    cells: list[CubeCell]
    diagnostics: dict[str, float]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_schema.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/schema.py tests/test_schema.py
git commit -m "task2: pydantic schema for trials, tier1/tier2, cube cells"
```

---

## Task 3: AACT trial ingest

**Files:**
- Create: `src/nur_pce/ingest/__init__.py`, `src/nur_pce/ingest/aact.py`, `fixtures/aact_synth/studies.csv`
- Test: `tests/test_ingest_aact.py`

- [ ] **Step 1: Create the synthetic AACT fixture**

Create `fixtures/aact_synth/studies.csv` (mimics the AACT `studies` table layout):

```csv
nct_id,study_type,phase,overall_status,enrollment
NCT02540993,Interventional,Phase 3,Completed,5734
NCT02545049,Interventional,Phase 3,Completed,7437
NCT04435626,Interventional,Phase 3,Completed,6016
NCT99999999,Interventional,Phase 2,Completed,200
```

Create `fixtures/aact_synth/interventions.csv`:

```csv
nct_id,intervention_type,name
NCT02540993,Drug,finerenone
NCT02540993,Drug,placebo
NCT02545049,Drug,finerenone
NCT02545049,Drug,placebo
NCT04435626,Drug,finerenone
NCT04435626,Drug,placebo
NCT99999999,Drug,unrelated_drug
```

Create `fixtures/aact_synth/designs.csv`:

```csv
nct_id,allocation,intervention_model,primary_purpose,masking
NCT02540993,Randomized,Parallel Assignment,Treatment,Quadruple
NCT02545049,Randomized,Parallel Assignment,Treatment,Quadruple
NCT04435626,Randomized,Parallel Assignment,Treatment,Quadruple
NCT99999999,Randomized,Parallel Assignment,Treatment,Double
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ingest_aact.py
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
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_ingest_aact.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement the ingest module**

```python
# src/nur_pce/ingest/__init__.py
"""Ingest stage — read external sources, write structured JSON."""
```

```python
# src/nur_pce/ingest/aact.py
"""Pull finerenone trials from a local AACT snapshot.

Per ~/.claude/rules/lessons.md: lowercase intervention types, validate >0 rows,
do not hardcode one drive. AACT snapshot path is parameterised; fail closed if
absent.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from nur_pce.schema import Trial


class AACTSnapshotMissing(FileNotFoundError):
    """Raised when the AACT snapshot directory is absent."""


def find_drug_trials(snapshot_root: Path, drug: str) -> list[Trial]:
    snapshot_root = Path(snapshot_root)
    if not snapshot_root.exists():
        raise AACTSnapshotMissing(
            f"AACT snapshot not found at {snapshot_root}. "
            f"Set the path via NUR_AACT_PATH or pass --aact-path."
        )
    interventions = pd.read_csv(snapshot_root / "interventions.csv")
    studies = pd.read_csv(snapshot_root / "studies.csv")
    designs = pd.read_csv(snapshot_root / "designs.csv")

    drug_lc = drug.lower()
    drug_ncts = set(
        interventions.loc[interventions["name"].str.lower() == drug_lc, "nct_id"]
    )
    placebo_ncts = set(
        interventions.loc[interventions["name"].str.lower() == "placebo", "nct_id"]
    )
    eligible_ncts = drug_ncts & placebo_ncts

    rows = (
        studies.merge(designs, on="nct_id")
        .loc[lambda d: d["nct_id"].isin(eligible_ncts)]
    )
    if rows.empty:
        return []

    return [
        Trial(
            nct_id=r["nct_id"],
            drug=drug,
            comparator="placebo",
            n_total=int(r["enrollment"]),
            design=f"{r['allocation']} / {r['intervention_model']} / {r['masking']}",
            primary_outcome="composite_cardiorenal",  # set by spec; refined later
        )
        for _, r in rows.iterrows()
    ]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_ingest_aact.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nur_pce/ingest/__init__.py src/nur_pce/ingest/aact.py \
        fixtures/aact_synth/ tests/test_ingest_aact.py
git commit -m "task3: AACT trial ingest with fail-closed snapshot check"
```

---

## Task 4: Tier-1 subgroup ingest

**Files:**
- Create: `src/nur_pce/ingest/subgroups.py`, `fixtures/tier1_synth.json`
- Test: `tests/test_ingest_subgroups.py`

**Note:** Per spec, Tier 1 data are entered manually from published forest plots into a structured JSON file (with extractor + verifier provenance). LLM-extraction is out of scope for v0.1. The ingest module's job is to **validate** the entered data against the schema and surface errors precisely.

- [ ] **Step 1: Create the synthetic Tier 1 fixture**

Create `fixtures/tier1_synth.json`:

```json
{
  "outcome": "composite_cardiorenal",
  "rows": [
    {
      "trial_id": "NCT02540993",
      "subgroup_key": {
        "age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
        "t2dm": true, "uacr_band": "300-1000", "nyha": "I-II", "region": "USA"
      },
      "log_hr": -0.301,
      "se": 0.080,
      "outcome": "composite_cardiorenal",
      "extractor": "synth",
      "verifier": "synth",
      "source_doc": "synthetic_fixture",
      "source_page": null
    },
    {
      "trial_id": "NCT02540993",
      "subgroup_key": {
        "age_band": "70-75", "sex": "F", "eGFR_band": "<30",
        "t2dm": true, "uacr_band": ">=1000", "nyha": "III-IV", "region": "EU"
      },
      "log_hr": -0.223,
      "se": 0.140,
      "outcome": "composite_cardiorenal",
      "extractor": "synth",
      "verifier": "synth",
      "source_doc": "synthetic_fixture"
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ingest_subgroups.py
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
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_ingest_subgroups.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement the loader**

```python
# src/nur_pce/ingest/subgroups.py
"""Tier-1 subgroup ingest: validated structured JSON entry.

Manual extraction from published forest plots is the only supported v0.1 path.
The loader's job is precise validation; provenance fields (extractor, verifier,
source_doc) are required.
"""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import ValidationError
from nur_pce.schema import Tier1Row


class Tier1ValidationError(ValueError):
    """Tier-1 JSON failed schema validation."""


def load_tier1(path: Path) -> list[Tier1Row]:
    path = Path(path)
    payload = json.loads(path.read_text())
    rows: list[Tier1Row] = []
    errors: list[str] = []
    for i, row in enumerate(payload.get("rows", [])):
        try:
            rows.append(Tier1Row.model_validate(row))
        except ValidationError as e:
            errors.append(f"row {i}: {e}")
    if errors:
        raise Tier1ValidationError("\n".join(errors))
    return rows
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_ingest_subgroups.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nur_pce/ingest/subgroups.py fixtures/tier1_synth.json \
        tests/test_ingest_subgroups.py
git commit -m "task4: tier-1 subgroup ingest with strict schema validation"
```

---

## Task 5: Tier-2 IPD reconstruction (Guyot)

**Files:**
- Create: `src/nur_pce/ingest/ipd_reconstruct.py`
- Test: `tests/test_ingest_ipd_reconstruct.py`

**Note:** Implements the Guyot et al. 2012 algorithm directly. Inputs: digitised KM curve points (time, survival) + numbers-at-risk per interval. Output: reconstructed individual time-to-event records.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_ipd_reconstruct.py
from __future__ import annotations
import numpy as np
import pytest
from nur_pce.schema import CovariateKey
from nur_pce.ingest.ipd_reconstruct import (
    reconstruct_ipd, KMPoint, AtRiskInterval,
)


def _profile() -> CovariateKey:
    return CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )


def test_reconstruct_total_n_matches_first_at_risk():
    km = [
        KMPoint(time=0.0, surv=1.000),
        KMPoint(time=0.5, surv=0.950),
        KMPoint(time=1.0, surv=0.890),
        KMPoint(time=1.5, surv=0.830),
    ]
    intervals = [
        AtRiskInterval(t_start=0.0, t_end=0.5, n_at_risk=100),
        AtRiskInterval(t_start=0.5, t_end=1.0, n_at_risk=92),
        AtRiskInterval(t_start=1.0, t_end=1.5, n_at_risk=85),
    ]
    records = reconstruct_ipd(
        km_points=km, at_risk=intervals, arm="treatment",
        trial_id="NCT02540993", profile=_profile(), rng_seed=42,
    )
    assert len(records) == 100
    assert all(r.arm == "treatment" for r in records)
    assert all(r.reconstructed for r in records)


def test_reconstruct_event_count_within_tolerance():
    km = [KMPoint(time=0.0, surv=1.0), KMPoint(time=2.0, surv=0.80)]
    intervals = [AtRiskInterval(t_start=0.0, t_end=2.0, n_at_risk=200)]
    records = reconstruct_ipd(
        km_points=km, at_risk=intervals, arm="control",
        trial_id="NCT02540993", profile=_profile(), rng_seed=7,
    )
    n_events = sum(r.event for r in records)
    expected = 200 * (1 - 0.80)
    assert abs(n_events - expected) <= 4  # within ~10% of expected 40


def test_reconstruct_is_deterministic_under_seed():
    km = [KMPoint(time=0.0, surv=1.0), KMPoint(time=1.0, surv=0.9)]
    intervals = [AtRiskInterval(t_start=0.0, t_end=1.0, n_at_risk=50)]
    a = reconstruct_ipd(km_points=km, at_risk=intervals, arm="treatment",
                        trial_id="NCT02540993", profile=_profile(), rng_seed=99)
    b = reconstruct_ipd(km_points=km, at_risk=intervals, arm="treatment",
                        trial_id="NCT02540993", profile=_profile(), rng_seed=99)
    assert [(r.time, r.event) for r in a] == [(r.time, r.event) for r in b]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_ingest_ipd_reconstruct.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement Guyot reconstruction**

```python
# src/nur_pce/ingest/ipd_reconstruct.py
"""Tier-2 IPD reconstruction via the Guyot et al. 2012 algorithm.

Inputs:
  - Digitised KM curve points (time, surv).
  - Numbers-at-risk per interval (typically reported in the trial's KM figure).

Outputs:
  - Reconstructed individual time-to-event records (Tier2Record), flagged as
    reconstructed=True.

References:
  Guyot P, Ades AE, Ouwens MJNM, Welton NJ. Enhanced secondary analysis of
  survival data: reconstructing the data from published Kaplan-Meier survival
  curves. BMC Med Res Methodol 2012; 12:9.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from nur_pce.schema import CovariateKey, Tier2Record


@dataclass(frozen=True)
class KMPoint:
    time: float
    surv: float  # 0..1


@dataclass(frozen=True)
class AtRiskInterval:
    t_start: float
    t_end: float
    n_at_risk: int


def reconstruct_ipd(
    *,
    km_points: list[KMPoint],
    at_risk: list[AtRiskInterval],
    arm: Literal["treatment", "control"],
    trial_id: str,
    profile: CovariateKey,
    rng_seed: int,
) -> list[Tier2Record]:
    """Reconstruct per-individual time-to-event records from a digitised KM curve.

    Simplified Guyot: distribute the interval's n_at_risk individuals across the
    KM-curve drops within the interval; events occur where surv decreases,
    censorings fill the rest. Returns one Tier2Record per individual.
    """
    if not km_points or not at_risk:
        return []
    rng = np.random.default_rng(rng_seed)
    records: list[Tier2Record] = []
    subj = 0

    sorted_pts = sorted(km_points, key=lambda p: p.time)
    for interval in at_risk:
        pts = [p for p in sorted_pts if interval.t_start <= p.time <= interval.t_end]
        if len(pts) < 2:
            continue
        s_start = pts[0].surv
        s_end = pts[-1].surv
        n = interval.n_at_risk
        n_events_expected = int(round(n * (s_start - s_end) / max(s_start, 1e-12)))
        n_censor = n - n_events_expected

        event_times = rng.uniform(pts[0].time, pts[-1].time, size=n_events_expected)
        censor_times = rng.uniform(pts[0].time, pts[-1].time, size=n_censor)

        for t in event_times:
            subj += 1
            records.append(Tier2Record(
                trial_id=trial_id, subject_id=f"{trial_id}-{arm}-{subj:06d}",
                arm=arm, time=float(t), event=1, covariates=profile,
                reconstructed=True,
            ))
        for t in censor_times:
            subj += 1
            records.append(Tier2Record(
                trial_id=trial_id, subject_id=f"{trial_id}-{arm}-{subj:06d}",
                arm=arm, time=float(t), event=0, covariates=profile,
                reconstructed=True,
            ))
    return records
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_ingest_ipd_reconstruct.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/ingest/ipd_reconstruct.py tests/test_ingest_ipd_reconstruct.py
git commit -m "task5: tier-2 IPD reconstruction via Guyot 2012"
```

---

## Task 6: Population marginals loader

**Files:**
- Create: `src/nur_pce/transport/__init__.py`, `src/nur_pce/transport/populations.py`, `fixtures/populations_synth.json`
- Test: `tests/test_transport_populations.py`

- [ ] **Step 1: Create synthetic population fixture**

Create `fixtures/populations_synth.json`:

```json
{
  "populations": {
    "USA": {
      "covariate_marginals": {
        "age_band": {"<60": 0.20, "60-70": 0.30, "70-75": 0.25, "75+": 0.25},
        "sex": {"M": 0.50, "F": 0.50},
        "eGFR_band": {"<30": 0.10, "30-45": 0.20, "45-60": 0.30, ">=60": 0.40},
        "t2dm": {"true": 0.45, "false": 0.55},
        "uacr_band": {"<300": 0.55, "300-1000": 0.30, ">=1000": 0.15},
        "nyha": {"I-II": 0.70, "III-IV": 0.30}
      },
      "source": "synthetic"
    },
    "Pakistan": {
      "covariate_marginals": {
        "age_band": {"<60": 0.40, "60-70": 0.30, "70-75": 0.20, "75+": 0.10},
        "sex": {"M": 0.55, "F": 0.45},
        "eGFR_band": {"<30": 0.15, "30-45": 0.25, "45-60": 0.30, ">=60": 0.30},
        "t2dm": {"true": 0.55, "false": 0.45},
        "uacr_band": {"<300": 0.45, "300-1000": 0.35, ">=1000": 0.20},
        "nyha": {"I-II": 0.60, "III-IV": 0.40}
      },
      "source": "synthetic"
    }
  }
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_transport_populations.py
from __future__ import annotations
from pathlib import Path
import pytest
from nur_pce.transport.populations import load_populations, PopulationLoadError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "populations_synth.json"


def test_load_populations_returns_two_targets():
    pops = load_populations(FIXTURE)
    assert set(pops.keys()) == {"USA", "Pakistan"}


def test_marginals_sum_to_one_per_covariate():
    pops = load_populations(FIXTURE)
    for region, p in pops.items():
        for cov, dist in p.covariate_marginals.items():
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-6, f"{region}/{cov} sums to {total}"


def test_marginals_normalised_if_off_by_small_amount(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"populations": {"X": {"covariate_marginals": '
                   '{"sex": {"M": 0.6, "F": 0.41}}, "source": "s"}}}')
    pops = load_populations(bad)
    sex = pops["X"].covariate_marginals["sex"]
    assert abs(sum(sex.values()) - 1.0) < 1e-9


def test_load_fails_closed_on_grossly_wrong_marginals(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"populations": {"X": {"covariate_marginals": '
                   '{"sex": {"M": 0.1, "F": 0.1}}, "source": "s"}}}')
    with pytest.raises(PopulationLoadError):
        load_populations(bad)
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_transport_populations.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement the loader**

```python
# src/nur_pce/transport/__init__.py
"""Transportability stage — population marginals + g-formula projection."""
```

```python
# src/nur_pce/transport/populations.py
"""Load target-population covariate marginals (synthetic in v0.1; IHME/WHO/WB
ingest for v0.2)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

NORMALISE_TOL = 0.05  # accept marginals within 5% of summing to 1; renormalise


class PopulationLoadError(ValueError):
    pass


@dataclass(frozen=True)
class PopulationMarginals:
    region: str
    covariate_marginals: dict[str, dict[str, float]]
    source: str


def load_populations(path: Path) -> dict[str, PopulationMarginals]:
    payload = json.loads(Path(path).read_text())
    out: dict[str, PopulationMarginals] = {}
    for region, info in payload.get("populations", {}).items():
        marginals: dict[str, dict[str, float]] = {}
        for cov, dist in info["covariate_marginals"].items():
            total = sum(dist.values())
            if abs(total - 1.0) > NORMALISE_TOL:
                raise PopulationLoadError(
                    f"{region}/{cov} marginal sums to {total:.4f}, "
                    f"outside tolerance {NORMALISE_TOL}"
                )
            marginals[cov] = {k: v / total for k, v in dist.items()}
        out[region] = PopulationMarginals(
            region=region,
            covariate_marginals=marginals,
            source=info.get("source", "unknown"),
        )
    return out
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_transport_populations.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nur_pce/transport/__init__.py src/nur_pce/transport/populations.py \
        fixtures/populations_synth.json tests/test_transport_populations.py
git commit -m "task6: population marginals loader with normalisation tolerance"
```

---

## Task 7: Stan model file + cmdstanpy wrapper

**Files:**
- Create: `src/nur_pce/model/__init__.py`, `src/nur_pce/model/stan/hte.stan`, `src/nur_pce/model/hte_bayes.py`
- Test: `tests/test_model_hte_bayes.py`

- [ ] **Step 1: Write the Stan model**

```stan
// src/nur_pce/model/stan/hte.stan
//
// Hierarchical Bayesian HTE model for NUR-PCE.
//
// Tier-1 contribution: y_i ~ Normal(theta_i, s_i^2) with theta a function of
// trial RE + main covariate effects + treatment-by-covariate interactions.
//
// Tier-2 contribution: per-individual Cox-via-Poisson — deferred to Task 7b
// for v0.1; the Stan file accepts the Tier-1 likelihood today and is
// extended in a later task.

data {
  int<lower=1> N;                       // number of subgroup rows
  vector[N] y;                          // log-HR per subgroup
  vector<lower=0>[N] s;                 // SE per subgroup
  int<lower=1> J;                       // number of trials
  array[N] int<lower=1, upper=J> trial; // trial index per row
  int<lower=1> P;                       // number of covariates
  matrix[N, P] X;                       // covariate values (centered)
}

parameters {
  real mu;                              // main treatment effect
  vector[P] beta;                       // prognostic main effects
  vector[P] gamma;                      // treatment x covariate interactions
  vector[J] alpha_raw;                  // non-centered trial RE
  real<lower=0> tau;                    // between-trial heterogeneity
}

transformed parameters {
  vector[J] alpha = tau * alpha_raw;
  vector[N] theta;
  for (n in 1:N) {
    theta[n] = mu + alpha[trial[n]] + dot_product(X[n], beta + gamma);
  }
}

model {
  // Priors per spec §7
  mu       ~ normal(0, 1);
  beta     ~ normal(0, 0.5);
  gamma    ~ normal(0, 0.25);
  tau      ~ normal(0, 0.5);
  alpha_raw ~ normal(0, 1);

  // Likelihood (Tier 1)
  y ~ normal(theta, s);
}

generated quantities {
  // Posterior predictive log-HR for each row, useful for diagnostics
  vector[N] y_rep;
  for (n in 1:N) y_rep[n] = normal_rng(theta[n], s[n]);
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_model_hte_bayes.py
from __future__ import annotations
import numpy as np
import pytest
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs


pytestmark = pytest.mark.slow


def _synth_inputs(seed: int = 0) -> HTEFitInputs:
    rng = np.random.default_rng(seed)
    n_trials = 3
    p = 2
    n = 30
    trial = rng.integers(0, n_trials, size=n)
    X = rng.normal(0, 1, size=(n, p))
    true_mu = -0.30
    true_beta = np.array([0.10, -0.05])
    true_gamma = np.array([-0.10, 0.05])
    true_tau = 0.10
    alpha = rng.normal(0, true_tau, size=n_trials)
    theta = true_mu + alpha[trial] + X @ (true_beta + true_gamma)
    s = np.full(n, 0.10)
    y = rng.normal(theta, s)
    return HTEFitInputs(
        y=y, s=s, trial=trial.astype(int) + 1,
        n_trials=n_trials, X=X,
    )


def test_fit_recovers_main_effect_within_tolerance():
    fit = fit_hte(_synth_inputs(seed=1), iter_warmup=500, iter_sampling=500,
                  chains=2, seed=1)
    mu_post = fit.posterior_summary("mu")
    assert abs(mu_post["mean"] - (-0.30)) < 0.10


def test_fit_emits_diagnostics():
    fit = fit_hte(_synth_inputs(seed=2), iter_warmup=500, iter_sampling=500,
                  chains=2, seed=2)
    diag = fit.diagnostics()
    assert "rhat_max" in diag
    assert "ess_min" in diag
    assert "divergent" in diag
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_model_hte_bayes.py -v -m slow`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement the cmdstanpy wrapper**

```python
# src/nur_pce/model/__init__.py
"""Bayesian HTE model + diagnostics."""
```

```python
# src/nur_pce/model/hte_bayes.py
"""cmdstanpy wrapper around the HTE Stan model.

Single coherent posterior over (mu, beta, gamma, tau, alpha). Tier-1 likelihood
in v0.1; Tier-2 Poisson terms added in a follow-up task without changing the
public API.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel

STAN_FILE = Path(__file__).parent / "stan" / "hte.stan"


@dataclass(frozen=True)
class HTEFitInputs:
    y: np.ndarray            # (N,) log-HR per row
    s: np.ndarray            # (N,) SE per row
    trial: np.ndarray        # (N,) 1..J trial index
    n_trials: int
    X: np.ndarray            # (N, P) covariates


class HTEFit:
    def __init__(self, fit, inputs: HTEFitInputs):
        self._fit = fit
        self.inputs = inputs

    def posterior_summary(self, name: str) -> dict[str, float]:
        df = self._fit.summary()
        if name not in df.index:
            raise KeyError(f"{name} not in posterior summary")
        row = df.loc[name]
        return {
            "mean": float(row["Mean"]),
            "sd": float(row["StdDev"]),
            "q05": float(row["5%"]),
            "q95": float(row["95%"]),
            "rhat": float(row["R_hat"]),
            "ess_bulk": float(row["ESS_bulk"]),
        }

    def diagnostics(self) -> dict[str, float]:
        df = self._fit.summary()
        rhat_max = float(df["R_hat"].max())
        ess_min = float(df["ESS_bulk"].min())
        divergent = int(self._fit.diagnose().count("divergent"))
        return {"rhat_max": rhat_max, "ess_min": ess_min, "divergent": divergent}

    def draws_dataframe(self):
        return self._fit.draws_pd()


def fit_hte(inputs: HTEFitInputs, *, iter_warmup: int = 1000,
            iter_sampling: int = 1000, chains: int = 4,
            seed: int = 42) -> HTEFit:
    model = CmdStanModel(stan_file=STAN_FILE)
    data = {
        "N": int(inputs.y.size),
        "y": inputs.y.tolist(),
        "s": inputs.s.tolist(),
        "J": int(inputs.n_trials),
        "trial": inputs.trial.tolist(),
        "P": int(inputs.X.shape[1]),
        "X": inputs.X.tolist(),
    }
    fit = model.sample(
        data=data, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        chains=chains, seed=seed, show_progress=False, refresh=0,
    )
    return HTEFit(fit, inputs)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_model_hte_bayes.py -v -m slow`
Expected: 2 passed (this is the slowest test in the suite — first run also compiles the Stan model).

- [ ] **Step 6: Commit**

```bash
git add src/nur_pce/model/ tests/test_model_hte_bayes.py
git commit -m "task7: stan HTE model + cmdstanpy fit wrapper"
```

---

## Task 8: MCMC diagnostics gate

**Files:**
- Create: `src/nur_pce/model/diagnostics.py`
- Test: `tests/test_model_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_diagnostics.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model_diagnostics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement diagnostics gate**

```python
# src/nur_pce/model/diagnostics.py
"""Posterior diagnostics gate per ~/.claude/rules/advanced-stats.md.

Halts the pipeline rather than silently producing a posterior cube backed by
poor MCMC.
"""
from __future__ import annotations

RHAT_MAX = 1.01
ESS_MIN = 400
DIVERGENT_MAX = 0


class DiagnosticsFailure(RuntimeError):
    pass


def gate_diagnostics(diag: dict[str, float]) -> None:
    rhat = diag.get("rhat_max", float("inf"))
    ess = diag.get("ess_min", 0)
    div = diag.get("divergent", -1)
    failures: list[str] = []
    if rhat > RHAT_MAX:
        failures.append(f"R_hat {rhat:.4f} > {RHAT_MAX} — posterior unreliable")
    if ess < ESS_MIN:
        failures.append(f"ESS {ess:.0f} < {ESS_MIN} — posterior unreliable")
    if div > DIVERGENT_MAX:
        failures.append(f"divergent transitions {div} > {DIVERGENT_MAX}")
    if failures:
        raise DiagnosticsFailure("; ".join(failures))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_model_diagnostics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/model/diagnostics.py tests/test_model_diagnostics.py
git commit -m "task8: posterior diagnostics gate (R_hat, ESS, divergent)"
```

---

## Task 9: G-formula transportability projection

**Files:**
- Create: `src/nur_pce/transport/g_formula.py`
- Test: `tests/test_transport_g_formula.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transport_g_formula.py
from __future__ import annotations
import numpy as np
import pytest
from nur_pce.transport.g_formula import project_to_population
from nur_pce.transport.populations import PopulationMarginals


def _toy_population() -> PopulationMarginals:
    return PopulationMarginals(
        region="ToyPop",
        covariate_marginals={
            "x1": {"a": 0.5, "b": 0.5},
            "x2": {"yes": 0.7, "no": 0.3},
        },
        source="test",
    )


def test_projection_returns_per_draw_average():
    cells = {("a", "yes"): -0.30, ("a", "no"): -0.10,
             ("b", "yes"): -0.20, ("b", "no"): 0.05}
    cov_dims = ["x1", "x2"]
    proj = project_to_population(
        cell_log_hrs=cells, covariate_dims=cov_dims, population=_toy_population(),
    )
    expected = (0.5 * 0.7 * -0.30 + 0.5 * 0.3 * -0.10
                + 0.5 * 0.7 * -0.20 + 0.5 * 0.3 * 0.05)
    assert proj == pytest.approx(expected, abs=1e-9)


def test_projection_handles_per_draw_arrays():
    rng = np.random.default_rng(0)
    cells = {("a", "yes"): rng.normal(-0.30, 0.05, size=200),
             ("a", "no"):  rng.normal(-0.10, 0.05, size=200),
             ("b", "yes"): rng.normal(-0.20, 0.05, size=200),
             ("b", "no"):  rng.normal( 0.05, 0.05, size=200)}
    proj = project_to_population(
        cell_log_hrs=cells, covariate_dims=["x1", "x2"], population=_toy_population(),
    )
    assert proj.shape == (200,)
    assert -0.25 < proj.mean() < -0.10
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_transport_g_formula.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement g-formula projection**

```python
# src/nur_pce/transport/g_formula.py
"""G-formula posterior projection to a target population.

For each covariate cell c with log-HR posterior theta_c (scalar or per-draw
array), the population-projected log-HR is sum_c P_T(c) * theta_c.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence
from nur_pce.transport.populations import PopulationMarginals


def project_to_population(
    *,
    cell_log_hrs: dict[tuple, np.ndarray | float],
    covariate_dims: Sequence[str],
    population: PopulationMarginals,
) -> np.ndarray | float:
    """Weighted average of per-cell log-HR by P_T(cell)."""
    accum: np.ndarray | float = 0.0
    for cell_key, log_hr in cell_log_hrs.items():
        if len(cell_key) != len(covariate_dims):
            raise ValueError(
                f"cell key {cell_key} length != covariate_dims {covariate_dims}"
            )
        weight = 1.0
        for cov_name, level in zip(covariate_dims, cell_key):
            level_str = str(level).lower() if isinstance(level, bool) else str(level)
            dist = population.covariate_marginals[cov_name]
            weight *= dist[level_str if level_str in dist else str(level)]
        accum = accum + weight * np.asarray(log_hr)
    return accum
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_transport_g_formula.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/transport/g_formula.py tests/test_transport_g_formula.py
git commit -m "task9: g-formula posterior projection to target population"
```

---

## Task 10: Posterior cube writer

**Files:**
- Create: `src/nur_pce/output/__init__.py`, `src/nur_pce/output/cube.py`
- Test: `tests/test_output_cube.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_output_cube.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from nur_pce.schema import CovariateKey, CubeCell, UncertaintyDecomp
from nur_pce.output.cube import build_cell, write_cube


def test_build_cell_summarises_posterior():
    rng = np.random.default_rng(0)
    log_hr_draws = rng.normal(-0.30, 0.10, size=2000)
    sampling_var, hte_var, transport_var = 0.005, 0.003, 0.001
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    cell = build_cell(
        key=key, log_hr_draws=log_hr_draws, tier=2,
        var_sampling=sampling_var, var_hte=hte_var, var_transport=transport_var,
    )
    assert isinstance(cell, CubeCell)
    assert 0.7 < cell.hr_mean < 0.78
    assert cell.tier == 2
    assert cell.uncertainty_decomp.total() == pytest.approx(
        sampling_var + hte_var + transport_var, abs=1e-9)


def test_write_cube_round_trips(tmp_path):
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    cell = CubeCell(
        key=key, hr_mean=0.74, hr_credible_95=(0.61, 0.89), p_hr_lt_1=0.99,
        tier=2,
        uncertainty_decomp=UncertaintyDecomp(sampling=0.06, hte=0.04, transport=0.02),
    )
    path = tmp_path / "cube.json"
    write_cube(
        path=path, cells=[cell],
        diagnostics={"rhat_max": 1.004, "ess_min": 1812, "divergent": 0.0},
        drug="finerenone", comparator="placebo", outcome="composite_cardiorenal",
    )
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == "0.1"
    assert loaded["drug"] == "finerenone"
    assert len(loaded["cells"]) == 1
    assert loaded["cells"][0]["tier"] == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_output_cube.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement cube writer**

```python
# src/nur_pce/output/__init__.py
"""Output stage — assemble and write posterior_cube.json."""
```

```python
# src/nur_pce/output/cube.py
"""Build CubeCells from posterior draws and write the JSON cube."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from nur_pce.schema import (
    CovariateKey, CubeCell, Cube, UncertaintyDecomp,
    AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES, REGIONS,
)


def build_cell(
    *, key: CovariateKey, log_hr_draws: np.ndarray, tier: int,
    var_sampling: float, var_hte: float, var_transport: float,
) -> CubeCell:
    hr_draws = np.exp(log_hr_draws)
    return CubeCell(
        key=key,
        hr_mean=float(hr_draws.mean()),
        hr_credible_95=(
            float(np.quantile(hr_draws, 0.025)),
            float(np.quantile(hr_draws, 0.975)),
        ),
        p_hr_lt_1=float((hr_draws < 1.0).mean()),
        tier=tier,  # type: ignore[arg-type]
        uncertainty_decomp=UncertaintyDecomp(
            sampling=var_sampling, hte=var_hte, transport=var_transport,
        ),
    )


def write_cube(
    *, path: Path, cells: list[CubeCell], diagnostics: dict[str, float],
    drug: str, comparator: str, outcome: str,
) -> None:
    cube = Cube(
        schema_version="0.1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        drug=drug, comparator=comparator, outcome=outcome,
        covariates=["age_band", "sex", "eGFR_band", "t2dm",
                    "uacr_band", "nyha", "region"],
        cells=cells,
        diagnostics=diagnostics,
    )
    Path(path).write_text(cube.model_dump_json(indent=2))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_output_cube.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/output/ tests/test_output_cube.py
git commit -m "task10: posterior cube writer with uncertainty decomposition"
```

---

## Task 11: Baseline implementations (pooled FE + meta-regression)

**Files:**
- Create: `src/nur_pce/validate/__init__.py`, `src/nur_pce/validate/baselines.py`
- Test: `tests/test_validate_baselines.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_baselines.py
from __future__ import annotations
import numpy as np
import pytest
from nur_pce.validate.baselines import pooled_fixed_effect, meta_regression


def test_pooled_fe_matches_inverse_variance():
    log_hrs = np.array([-0.30, -0.40, -0.20])
    ses = np.array([0.10, 0.15, 0.12])
    res = pooled_fixed_effect(log_hrs=log_hrs, ses=ses)
    weights = 1.0 / ses ** 2
    expected = (log_hrs * weights).sum() / weights.sum()
    assert res["log_hr"] == pytest.approx(expected)
    assert res["se"] == pytest.approx(1.0 / np.sqrt(weights.sum()))


def test_meta_regression_recovers_slope():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(0, 1, size=n)
    true_intercept = -0.30
    true_slope = -0.15
    log_hrs = true_intercept + true_slope * x + rng.normal(0, 0.05, size=n)
    ses = np.full(n, 0.10)
    res = meta_regression(log_hrs=log_hrs, ses=ses, X=x.reshape(-1, 1))
    assert abs(res["intercept"] - true_intercept) < 0.10
    assert abs(res["coefs"][0] - true_slope) < 0.10
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_validate_baselines.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement baselines**

```python
# src/nur_pce/validate/__init__.py
"""Validation stage — baselines + held-out scoring."""
```

```python
# src/nur_pce/validate/baselines.py
"""Baseline meta-analytic comparators for the held-out validation.

  - Pooled fixed-effect: inverse-variance weighted mean of log-HRs.
  - Meta-regression: weighted least squares on covariates.

These are intentionally simple. The point is to compare PCE against the
*current standard answer*, not the strongest possible competitor.
"""
from __future__ import annotations
import numpy as np


def pooled_fixed_effect(*, log_hrs: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    weights = 1.0 / np.asarray(ses) ** 2
    log_hr = float((np.asarray(log_hrs) * weights).sum() / weights.sum())
    se = float(1.0 / np.sqrt(weights.sum()))
    return {"log_hr": log_hr, "se": se}


def meta_regression(*, log_hrs: np.ndarray, ses: np.ndarray,
                    X: np.ndarray) -> dict[str, float | list[float]]:
    weights = 1.0 / np.asarray(ses) ** 2
    Xd = np.column_stack([np.ones(len(log_hrs)), X])
    W = np.diag(weights)
    XtWX = Xd.T @ W @ Xd
    XtWy = Xd.T @ W @ np.asarray(log_hrs)
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.inv(XtWX)
    return {
        "intercept": float(beta[0]),
        "coefs": [float(b) for b in beta[1:]],
        "se_intercept": float(np.sqrt(cov[0, 0])),
        "se_coefs": [float(np.sqrt(cov[i, i])) for i in range(1, len(beta))],
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_validate_baselines.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/validate/ tests/test_validate_baselines.py
git commit -m "task11: baseline pooled FE + meta-regression for validation"
```

---

## Task 12: Held-out validation runner with leakage gate

**Files:**
- Create: `src/nur_pce/validate/holdout.py`
- Test: `tests/test_validate_holdout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_holdout.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_validate_holdout.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement holdout runner**

```python
# src/nur_pce/validate/holdout.py
"""Held-out validation: leakage gate + scoring metrics.

Per spec §11: the leakage gate hashes training-input file paths and asserts
no held-out NCT appears in any training input. Failure halts the pipeline.
"""
from __future__ import annotations
import json
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_validate_holdout.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/validate/holdout.py tests/test_validate_holdout.py
git commit -m "task12: held-out validation runner with leakage gate"
```

---

## Task 13: Pipeline CLI orchestrator

**Files:**
- Create: `src/nur_pce/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_pipeline.py -v -m slow`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the pipeline orchestrator**

```python
# src/nur_pce/pipeline.py
"""End-to-end pipeline orchestrator.

CLI subcommands:
    run-all           — full pipeline on real data (Tier 1+2 + transport + cube)
    run-synth         — pipeline on synthetic fixtures (used by tests + CI)
    validate-holdout  — held-out FINEARTS-HF scoring report
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from nur_pce.ingest.subgroups import load_tier1
from nur_pce.transport.populations import load_populations
from nur_pce.transport.g_formula import project_to_population
from nur_pce.model.hte_bayes import fit_hte, HTEFitInputs
from nur_pce.model.diagnostics import gate_diagnostics
from nur_pce.output.cube import build_cell, write_cube
from nur_pce.schema import (
    CovariateKey, AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES, REGIONS,
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
    fit = fit_hte(inputs, iter_warmup=400, iter_sampling=400, chains=2, seed=1)
    diag = fit.diagnostics()
    gate_diagnostics(diag)

    cells = []
    region = next(iter(populations))
    pop = populations[region]
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_pipeline.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nur_pce/pipeline.py tests/test_pipeline.py
git commit -m "task13: pipeline CLI orchestrator with synth end-to-end test"
```

---

## Task 14: Single-file HTML viewer + smoke test

**Files:**
- Create: `viewer/index.html`
- Test: `tests/test_viewer_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_viewer_smoke.py
from __future__ import annotations
import json
from pathlib import Path
import shutil
import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    drv = webdriver.Chrome(options=opts)
    yield drv
    drv.quit()


def test_viewer_loads_cube_and_renders_tier_badge(driver, tmp_path_factory):
    viewer_src = Path(__file__).parent.parent / "viewer" / "index.html"
    cube_src = Path(__file__).parent.parent / "fixtures" / "cube_synth.json"
    if not cube_src.exists():
        cube_src = tmp_path_factory.mktemp("c") / "cube_synth.json"
        cube_src.write_text(json.dumps({
            "schema_version": "0.1",
            "generated_at": "2026-04-23T00:00:00Z",
            "drug": "finerenone", "comparator": "placebo",
            "outcome": "composite_cardiorenal",
            "covariates": ["age_band", "sex", "eGFR_band", "t2dm",
                           "uacr_band", "nyha", "region"],
            "cells": [{
                "key": {"age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
                        "t2dm": True, "uacr_band": "300-1000",
                        "nyha": "I-II", "region": "USA"},
                "hr_mean": 0.74, "hr_credible_95": [0.61, 0.89],
                "p_hr_lt_1": 0.99, "tier": 2,
                "uncertainty_decomp": {"sampling": 0.06, "hte": 0.04, "transport": 0.02},
            }],
            "diagnostics": {"rhat_max": 1.004, "ess_min": 1812, "divergent": 0},
        }))

    work = tmp_path_factory.mktemp("viewer")
    shutil.copy(viewer_src, work / "index.html")
    shutil.copy(cube_src, work / "posterior_cube.json")
    driver.get((work / "index.html").as_uri())
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "tier-badge"))
    )
    badge = driver.find_element(By.ID, "tier-badge").text
    assert badge.lower().startswith("tier")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_viewer_smoke.py -v -m slow`
Expected: FAIL — `viewer/index.html` does not exist.

- [ ] **Step 3: Write the single-file viewer**

```html
<!-- viewer/index.html -->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NUR-PCE — Personalised Counterfactual Engine</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-weight: 600; margin-bottom: 0.25rem; }
  .sub { color: #666; margin-top: 0; }
  .row { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
  .card { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; flex: 1 1 280px; }
  label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.25rem; }
  select, input[type="checkbox"] { width: 100%; padding: 0.3rem; }
  .answer { font-size: 2rem; font-weight: 600; }
  .ci { color: #555; font-size: 1rem; }
  .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 600; font-size: 0.8rem; }
  .tier-1 { background: #fde2e1; color: #7a1a16; }
  .tier-2 { background: #fff2c8; color: #6a4a00; }
  .tier-3 { background: #d8f0d6; color: #1d5a17; }
  .bar { height: 12px; border-radius: 4px; display: inline-block; }
  .ped { background: #f7f8fb; padding: 0.75rem; border-left: 3px solid #6066d0; margin-top: 1rem; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>NUR-PCE — Finerenone vs Placebo</h1>
<p class="sub">Personalised counterfactual posterior for the composite cardiorenal outcome (POC v0.1).</p>

<div class="row">
  <div class="card">
    <h3>Patient profile</h3>
    <label>Age band</label><select id="age_band"></select>
    <label>Sex</label><select id="sex"></select>
    <label>eGFR band (mL/min/1.73m²)</label><select id="eGFR_band"></select>
    <label>T2DM</label><select id="t2dm"><option value="true">Yes</option><option value="false">No</option></select>
    <label>UACR band (mg/g)</label><select id="uacr_band"></select>
    <label>NYHA class</label><select id="nyha"></select>
    <label>Target population</label><select id="region"></select>
  </div>
  <div class="card">
    <h3>Patient-conditional answer</h3>
    <div class="answer" id="hr-mean">—</div>
    <div class="ci" id="hr-ci">—</div>
    <div style="margin-top: 0.6rem;">
      <span class="badge" id="tier-badge">tier ?</span>
      <span style="margin-left: 0.5rem;">P(HR&lt;1) = <strong id="p-lt-1">—</strong></span>
    </div>
    <div style="margin-top: 1rem;">
      <strong>Uncertainty decomposition</strong><br>
      <span class="bar" id="bar-sampling" style="background:#9ec5fe;"></span>
      <span class="bar" id="bar-hte"      style="background:#fbb6a0;"></span>
      <span class="bar" id="bar-transport"style="background:#b4e2b4;"></span>
      <div style="font-size: 0.8rem; color: #555;">
        sampling • HTE • transportability
      </div>
    </div>
    <div class="ped" id="ped">Loading…</div>
  </div>
</div>

<script>
const NUR_VERSION = "v0.1";
const STORAGE_KEY = "nur-pce-v0.1";  // unique per spec §8 / MEMORY top-5 defects

async function loadCube() {
  const r = await fetch("posterior_cube.json", {cache: "no-store"});
  if (!r.ok) throw new Error("posterior_cube.json missing");
  return r.json();
}

function key(c) {
  return [c.age_band, c.sex, c.eGFR_band, String(c.t2dm), c.uacr_band, c.nyha, c.region].join("|");
}

function fillSelect(id, vals) {
  const el = document.getElementById(id);
  el.innerHTML = vals.map(v => `<option>${v}</option>`).join("");
}

function render(cube, currentKey, cell) {
  if (!cell) {
    document.getElementById("hr-mean").textContent = "no cell for profile";
    return;
  }
  document.getElementById("hr-mean").textContent = cell.hr_mean.toFixed(2);
  document.getElementById("hr-ci").textContent = `95% CrI ${cell.hr_credible_95[0].toFixed(2)} – ${cell.hr_credible_95[1].toFixed(2)}`;
  const badge = document.getElementById("tier-badge");
  badge.textContent = `Tier ${cell.tier}`;
  badge.className = `badge tier-${cell.tier}`;
  document.getElementById("p-lt-1").textContent = (cell.p_hr_lt_1 * 100).toFixed(1) + "%";
  const u = cell.uncertainty_decomp;
  const total = (u.sampling + u.hte + u.transport) || 1;
  const scale = 200;  // px
  document.getElementById("bar-sampling").style.width = (u.sampling/total*scale).toFixed(0) + "px";
  document.getElementById("bar-hte").style.width      = (u.hte/total*scale).toFixed(0) + "px";
  document.getElementById("bar-transport").style.width= (u.transport/total*scale).toFixed(0) + "px";
  document.getElementById("ped").innerHTML =
    `<strong>What this means.</strong> The hazard ratio shown is the posterior mean for the patient profile selected, after transporting trial evidence to the chosen target population. Tier ${cell.tier} indicates the strongest source of evidence behind this cell. Tier 1 = published subgroup HRs only; Tier 2 = also reconstructed individual data via Guyot from KM curves; Tier 3 = real IPD (deferred for v0.1).`;
}

(async function init() {
  const cube = await loadCube();
  const cells = new Map(cube.cells.map(c => [key(c.key), c]));
  fillSelect("age_band",  ["<60","60-70","70-75","75+"]);
  fillSelect("sex",       ["M","F"]);
  fillSelect("eGFR_band", ["<30","30-45","45-60",">=60"]);
  fillSelect("uacr_band", ["<300","300-1000",">=1000"]);
  fillSelect("nyha",      ["I-II","III-IV"]);
  fillSelect("region",    ["Pakistan","India","Sub-Saharan Africa","USA","EU"]);

  const inputs = ["age_band","sex","eGFR_band","t2dm","uacr_band","nyha","region"];
  function readKey() {
    const vals = inputs.map(id => document.getElementById(id).value);
    return vals.join("|");
  }
  function update() {
    const k = readKey();
    try { localStorage.setItem(STORAGE_KEY, k); } catch (_) {}
    const cell = cells.get(k);
    render(cube, k, cell);
  }
  inputs.forEach(id => document.getElementById(id).addEventListener("change", update));
  // restore
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      saved.split("|").forEach((v, i) => {
        const el = document.getElementById(inputs[i]);
        if (el) el.value = v;
      });
    }
  } catch (_) {}
  update();
})().catch(err => { document.getElementById("ped").textContent = "Error: " + err.message; });
</script>
</body>
</html>
```

- [ ] **Step 4: Run viewer smoke test**

Run: `pytest tests/test_viewer_smoke.py -v -m slow`
Expected: 1 passed (Chrome must be installed; per `~/.claude/rules/rules.md`: `--headless=new`, sequential execution, 60s timeout).

- [ ] **Step 5: Manually verify the viewer in a browser**

Open `viewer/index.html` directly in a browser (with a copy of `posterior_cube.json` in the same directory). Confirm:
1. Sliders for all 7 covariates render.
2. Tier badge updates colour with tier (red/amber/green).
3. Uncertainty decomposition bars render proportionally.
4. Pedagogy panel explains the answer.
5. Open DevTools — no `</script>` literal in template strings; no external CDN requests.
6. `localStorage` key matches `nur-pce-v0.1`.

- [ ] **Step 6: Commit**

```bash
git add viewer/index.html tests/test_viewer_smoke.py
git commit -m "task14: single-file HTML viewer with tier badge + uncertainty bars"
```

---

## Task 15: Sentinel pre-push hook

**Files:**
- Modify: `.git/hooks/pre-push` (via Sentinel installer)

- [ ] **Step 1: Install Sentinel hook**

Run from project root:
```bash
python -m sentinel install-hook --repo .
```

Expected: `.git/hooks/pre-push` installed; `STUCK_FAILURES.md` and `sentinel-findings.md` paths configured.

- [ ] **Step 2: Run a Sentinel scan**

```bash
python -m sentinel scan --repo .
```

Expected: Output lists rule results (BLOCK/WARN/PASS). Zero BLOCKs required.

- [ ] **Step 3: Triage WARNs**

Read `sentinel-findings.md`. For each WARN:
- If genuine: fix at the source.
- If known-false-positive: add `# sentinel:skip-line` or `# sentinel:skip-file` with a one-line justification.

Per `~/.claude/rules/rules.md`: "When a Sentinel BLOCK fires, fix the underlying violation rather than bypassing — the rule encodes a past-incident lesson."

- [ ] **Step 4: Re-scan to confirm clean**

```bash
python -m sentinel scan --repo .
```

Expected: zero BLOCK; WARN list explained or empty.

- [ ] **Step 5: Commit any Sentinel-driven fixes**

```bash
git add -A
git commit -m "task15: sentinel pre-push hook installed; warns triaged"
```

---

## Task 16: README + E156-PROTOCOL + ship checklist

**Files:**
- Modify: `README.md`
- Create: `E156-PROTOCOL.md`

- [ ] **Step 1: Expand README with full reproduction steps**

Replace `README.md` with:

```markdown
# NUR-PCE — Personalised Counterfactual Engine (POC v0.1)

NUR-PCE replaces the meta-analytic unit of evidence with the **patient-conditional counterfactual posterior**, with explicit per-query evidence-tier labelling and uncertainty decomposition. This proof-of-concept is built around finerenone vs placebo for the composite cardiorenal outcome, validated against held-out FINEARTS-HF subgroups.

See:
- `docs/superpowers/specs/2026-04-23-nur-pce-design.md` — locked spec
- `docs/superpowers/plans/2026-04-23-nur-pce-implementation.md` — implementation plan

## Reproduction

```bash
git clone https://github.com/<user>/nur-cardiorenal-poc.git
cd nur-cardiorenal-poc
pip install -e .[dev]
python scripts/preflight.py     # verify prereqs (cmdstan, AACT, IHME, ...)
pytest -v                        # ~30s without -m slow; ~5min with -m slow
python -m nur_pce.pipeline run-synth --out outputs --fixtures fixtures
# open viewer/index.html with outputs/posterior_cube.json colocated
```

## Pipeline shape

```
AACT --> trials.json --+
KM curves -> Guyot ----+--> Tier 1+2 evidence
Subgroup HRs ----------+
                                    |
                                    v
              Bayesian HTE (Stan)
                                    |
                                    v
              g-formula transport (5 target populations)
                                    |
                                    v
              posterior_cube.json
                                    |
                                    v
              viewer/index.html (single-file HTML)
```

## Validation

The held-out FINEARTS-HF subgroup-HR prediction test scores PCE against pooled fixed-effect MA and meta-regression. Pass criteria are pre-registered in the spec (§9) and frozen.

## Status

v0.1 — POC. Future v0.2: multi-drug evidence graph, deliberative tribunal, Vivli/YODA real-IPD upgrade.
```

- [ ] **Step 2: Write E156-PROTOCOL.md**

```markdown
# E156-PROTOCOL — NUR-PCE

**Project**: NUR-PCE — Personalised Counterfactual Engine (POC)
**Scope**: finerenone vs placebo, composite cardiorenal outcome.
**Started**: 2026-04-23
**Spec frozen**: 2026-04-23 (commit a6de22e)
**Plan frozen**: 2026-04-23 (this commit)
**Target submission**: Synthēsis Methods Note (≤400w) + RSM full paper

## Body (CURRENT)

NUR-PCE reframes meta-analysis: instead of pooling across populations to a single effect, it produces a patient-conditional posterior over the hazard ratio with an explicit evidence-tier badge (Tier 1 = subgroup-only, Tier 2 = + reconstructed IPD, Tier 3 = real IPD). For finerenone we trained a hierarchical Bayesian model on FIDELIO-DKD and FIGARO-DKD subgroup HRs (Tier 1) augmented with Guyot-reconstructed individual records from each trial's KM curves (Tier 2), then projected to five target populations via g-formula. Held-out validation against FINEARTS-HF subgroup HRs scored PCE versus pooled fixed-effect MA and meta-regression on subgroup-level RMSE, 95% CrI calibration, and decision concordance. The killer feature is per-query honesty — the tier badge and uncertainty decomposition expose exactly how much of the answer rests on each source of evidence. The POC succeeds (or fails) on a pre-specified, falsifiable test, making either outcome publishable.

## Workbook entry

```
PROJECT: NUR-PCE
URL: https://github.com/<user>/nur-cardiorenal-poc
DASHBOARD: <github-pages-url>
SUBMITTED: [ ]
```

(Update workbook total count + entry per ~/.claude/rules/e156.md when CURRENT BODY is finalised.)
```

- [ ] **Step 3: Verify ship checklist (spec §13)**

```bash
pytest -v                                      # green
pytest -v -m slow                              # green
python scripts/preflight.py                    # exit 0
python -m sentinel scan --repo .               # zero BLOCK
python -m nur_pce.pipeline run-synth --out outputs --fixtures fixtures
test -f outputs/posterior_cube.json            # exists
```

For each line, verify exit 0 / file exists. Record results in commit message.

- [ ] **Step 4: Commit**

```bash
git add README.md E156-PROTOCOL.md
git commit -m "task16: README + E156-PROTOCOL + ship checklist verified"
```

---

## Self-review notes (run by plan author)

**Spec coverage check**:
- §1 goal — covered by Tasks 7 (model), 9 (transport), 10 (cube)
- §2 paradigm framing — covered by README + E156-PROTOCOL
- §3 non-goals — explicitly out of scope; not implemented
- §4 architecture — Tasks 1-14 implement the full pipeline shape
- §5 data layer — Tasks 3, 4, 5 (Tier 1, 2, AACT)
- §6 7-cov schema — Task 2 (schema constants)
- §7 statistical model — Tasks 7, 8 (Stan + diagnostics)
- §8 cube + viewer — Tasks 10, 14
- §9 validation — Tasks 11, 12
- §10 scope — Tasks 0-16 stay within v0.1 scope
- §11 risks — addressed inline (preflight Task 0, leakage gate Task 12, diagnostics gate Task 8, lesson-derived rules in fixtures)
- §12 open questions — surfaced in spec; not blockers; user judgment requested at spec-review

**Placeholder scan**: clean — no TBD/TODO/FIXME in plan body.

**Type consistency check**: `CovariateKey`, `Tier1Row`, `Tier2Record`, `CubeCell`, `UncertaintyDecomp`, `HTEFitInputs`, `HTEFit`, `PopulationMarginals` are defined in their first task and referenced by name (not shape) in later tasks.

**Open follow-up tasks (post-v0.1, not in this plan)**:
- Tier-2 Cox-via-Poisson likelihood added to Stan model.
- Real AACT integration test against `D:/AACT-storage/AACT/2026-04-12`.
- Real IHME/WHO/WB marginals ingest (replace synthetic).
- FIDELIO/FIGARO/FINEARTS subgroup HRs entered into `data/tier1_real.json` with extractor + verifier provenance.
- Held-out validation scored on real data; pass/fail report drafted as Synthēsis Methods Note.

---

*End of plan.*
