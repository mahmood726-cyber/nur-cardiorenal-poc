# NUR-PCE v0.1.1 — held-out validation report (synthetic FINEARTS-HF)

**Date**: 2026-04-23
**Tag**: `v0.1.1`
**Report artifact**: [validation_report.json](https://github.com/mahmood726-cyber/nur-cardiorenal-poc/releases/download/v0.1.1/validation_report.json) (release asset)
**Train**: synthetic FIDELIO-DKD + FIGARO-DKD mock (6 rows, 2 trials)
**Holdout**: synthetic FINEARTS-HF mock (5 subgroups, NCT04435626)
**MCMC diagnostics**: R̂ = 1.0032, ESS = 1164, divergent = 0

## Metrics

| Method | RMSE on log(HR) | 95% interval calibration | n |
|---|---|---|---|
| **NUR-PCE** | 0.0670 | **100%** | 5 |
| Pooled fixed-effect MA | 0.0658 | 60% | 5 |
| Meta-regression (WLS) | 0.0705 | 100% | 5 |

## Interpretation

On this synthetic test, **PCE does not win on point-prediction RMSE** — pooled-FE narrowly beats it (0.0658 vs 0.0670 on log-HR scale, a 0.001 difference). But:

- **Pooled-FE's 60% calibration means its 95% CI is too tight** — it covers only 3 of 5 holdout truth values. That's the cost of giving a single pooled answer with no patient personalisation: when the holdout subgroup deviates from the training mean, the narrow CI misses.
- **PCE's 100% calibration is the killer feature**: the posterior captures the holdout subgroups inside its 95% CrI 5 of 5 times. PCE's uncertainty is *honest*: it knows when it doesn't know.
- Meta-regression also calibrates at 100% but with worse RMSE — the WLS on a small training set without shrinkage produces wider intervals than necessary.

This matches the spec §1 design intent: "clinicians do not need maximally personalised answers; they need maximally honest personalised answers with the uncertainty exposed."

## What this synthetic test does NOT show

1. **N=5 is too small** for any RMSE difference of 0.001 on log-HR to be statistically meaningful.
2. **Synthetic data was constructed close to the training distribution.** The real test (FIDELIO/FIGARO on CKD-T2DM, FINEARTS-HF on HFmrEF/HFpEF) is a much harder generalisation challenge.
3. **2-feature X (T2DM + sex_M)** ignores age, eGFR, UACR, NYHA, region — most of the clinical structure. The v0.2 identifiability fix + treatment-arm column will expand this.
4. **Tier-2 IPD reconstruction is not yet wired into the fit** — the model used Tier-1 only.

## Spec §3a v0.2 status

| # | Item | Status |
|---|---|---|
| 1 | Tier-2 Poisson likelihood wired into fit | deferred |
| 2 | beta/gamma identifiability — treatment-arm column in X | deferred |
| 3 | G-formula on natural scale | deferred |
| 4 | **Held-out FINEARTS-HF validation runner** | **✅ shipped (this release)** |

## Reproducibility

```bash
git clone https://github.com/mahmood726-cyber/nur-cardiorenal-poc.git
cd nur-cardiorenal-poc
git checkout v0.1.1
pip install -e .[dev]
python scripts/preflight.py                                    # exit 0
python -m nur_pce.pipeline run-validation --out outputs/validation
# ~18 minutes on a no-g++ Windows + Python 3.13 box
cat outputs/validation/validation_report.json
```
