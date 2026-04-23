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
