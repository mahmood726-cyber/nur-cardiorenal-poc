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
python scripts/preflight.py     # verify prereqs (pymc, AACT, IHME, ...)
pytest -v                        # ~5s without -m slow; ~20min with -m slow
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
              PyMC hierarchical HTE
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

## Engine note

The plan was amended at preflight to use **PyMC 5 as the primary Bayesian engine** (Stan/cmdstanpy not present on the target machine). PyMC fits are slower in pure-Python PyTensor mode but produce equivalent posteriors. Stan port is a future optimisation.
