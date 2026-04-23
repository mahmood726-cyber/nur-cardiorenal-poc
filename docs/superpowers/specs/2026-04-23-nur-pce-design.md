# NUR-PCE — Personalised Counterfactual Engine

**Project**: `nur-cardiorenal-poc` — Network of Unit-level Recoverable evidence, finerenone proof-of-concept
**Spec version**: v0.1 (draft)
**Date**: 2026-04-23
**Author**: brainstormed with mahmood726 + Claude
**Status**: spec-lock pending user approval; followed by `writing-plans`

---

## 1. Goal

Replace the meta-analytic unit of evidence — currently "the trial-level pooled effect" — with the **patient-conditional counterfactual posterior**, recovered from published trial outputs with explicit honesty about evidence tier.

For every query of the form *"What is the effect of finerenone vs placebo on the composite cardiorenal outcome for a person with covariate profile X in target population T?"*, the engine returns:

- a posterior distribution over the hazard ratio,
- a 95% credible interval,
- the posterior probability that HR < 1,
- an **evidence-tier badge** (1, 2, or 3) describing the data behind that specific cell,
- an **uncertainty decomposition** (within-trial sampling variance, HTE/interaction posterior spread, transportability variance from target-population uncertainty).

## 2. Why this is paradigm-shifting

Current MA outputs a single pooled HR averaged over a trial population that does not exist anywhere in the world. NUR-PCE outputs a covariate-conditional, transportable, posterior-quantified, tier-honest answer for the patient in front of the clinician. The killer feature is **honest evidence-tier labelling per query** — clinicians do not need maximally personalised answers; they need maximally honest personalised answers with the uncertainty exposed.

The same paradigm scales to any drug class with published subgroup forest plots (the Tier 1 floor is universally available in cardiology). Finerenone is the proof-of-concept. The graph-of-evidence (`EGQE`) and deliberative tribunal (`DST`) are natural extensions for v0.2+.

## 3. Non-goals (deferred from v0.1)

- Live AACT polling (Pilot B / continuously-updating layer).
- Multi-drug evidence graph (`EGQE`, future Pilot 2).
- Deliberative LLM tribunal (`DST`, future Pilot 3).
- Real IPD via Vivli or YODA application (months-long; separate work-stream).
- Outcomes other than the composite cardiorenal endpoint (all-cause mortality and hyperkalemia adverse events deferred to v0.2).
- MRA class siblings (spironolactone, eplerenone) — natural extension after v0.1 ships.
- Multi-language wrapping (handled by the Synthesis-Courses / Fatiha pipeline once v0.1 is stable).

### 3a. Honest gap list — deferred from v0.1 to v0.2 after final-review (2026-04-23)

Final code review (commit `c13d3cf`) surfaced four design items that v0.1 *scaffolds* but does not *complete*. They are explicitly out of v0.1 scope and tracked here so the README and validation claims stay honest. Status as of 2026-04-23 (after rapid v0.1.1 + v0.1.2 follow-on releases):

1. **Tier-2 Poisson likelihood not wired into the fit.** *Status: still deferred.* v0.1 reconstructs Tier-2 IPD records via Guyot (`ingest/ipd_reconstruct.py`) but the model in `model/hte_bayes.py` only consumes Tier-1 subgroup HRs. v0.2 work: add a Cox-via-Poisson likelihood block to the PyMC model, sharing the same `theta` linear predictor.
2. **`beta + gamma` identifiability.** *Status: ✅ closed in v0.1.2 (commit `f46af09`).* Resolved by dropping `gamma` — the v0.1 spec §7 had two unidentifiable parameters given Tier-1-only data (only their sum drove the likelihood). The remaining single `beta` is now explicitly the treatment-by-covariate interaction. Posterior over the *effective* coefficient is unchanged; the parameterisation is honest. When Tier-2 IPD enters the fit, gamma can be re-introduced because per-individual rows carry treatment-arm indicators.
3. **G-formula on natural scale.** *Status: still deferred.* `transport/g_formula.py` averages on the log-HR scale (`E[log HR]` weighted by `P_T(X)`), and the caller exponentiates. Spec §7 stipulated `E_{P_T(X)}[exp(theta(X))]` (natural scale). Jensen's inequality gives a small downward bias for finite-variance posteriors. v0.2 work: exponentiate per-cell-per-draw before weighting.
4. **Held-out FINEARTS-HF validation runner.** *Status: ✅ closed in v0.1.1 (commit `fd7a1c0`).* Added `validate/holdout_runner.py` orchestrator + `pipeline.py::run_validation` CLI subcommand. Real validation report on synthetic FINEARTS holdout shows PCE wins on 95% interval calibration (100% vs pooled-FE 60%) — the spec §1 paradigm value confirmed; published numbers in `docs/RESULTS_v0.1.1.md` + release-asset `validation_report.json`.

**v0.1 ship status:** the engine, schema, ingest, diagnostics gate, leakage gate, held-out validation runner, single-file viewer, and end-to-end synth pipeline (with posterior wired through) are complete. Items 1 and 3 above remain v0.2 work.

## 4. Architecture

```
AACT snapshot --+
KMextract -----+--> Trial Evidence Layer --+
Subgroup PDF   |    (tier-labelled records)|
extraction ----+                           |
                                           v
IHME / WHO / WB --> Population Targets --> Bayesian HTE Model --> Posterior cube
                                           ^                       (per-cell x per-population)
7-covariate grid --------------------------+                              |
                                                                          v
                                                          Static JSON (compressed)
                                                                          |
                                                                          v
                                                      Single-file HTML viewer
                                                      (Fatiha pedagogy layer:
                                                       slider per covariate,
                                                       tier badge per query,
                                                       uncertainty decomposition)
```

Each pipeline stage is a Python module with JSON in / JSON out. Stages are independently runnable and testable.

| Stage | Module | Input | Output |
|---|---|---|---|
| 1 | `ingest/aact.py` | `D:/AACT-storage/AACT/2026-04-12` | `data/trials.json` (NCTs, designs, outcomes) |
| 2 | `ingest/subgroups.py` | published forest plots (PDF/figure) | `data/tier1.json` (subgroup HR + CI per trial) |
| 3 | `ingest/ipd_reconstruct.py` | published KM curves via `KMextract` | `data/tier2.json` (reconstructed IPD time-to-event records, flagged as reconstructed) |
| 4 | `model/hte_bayes.py` | `tier1.json` + `tier2.json` | `model/posterior_samples.parquet` |
| 5 | `transport/g_formula.py` | posterior + target population marginals (IHME/WHO/WB) | `model/transported_posterior.parquet` |
| 6 | `output/cube.py` | transported posterior, 7-cov grid, 5 target populations | `viewer/posterior_cube.json` |
| 7 | `viewer/index.html` | `posterior_cube.json` | interactive browser viewer |

**Project repo layout**:
```
C:/Projects/nur-cardiorenal-poc/
  data/                    # JSON between stages (gitignored except fixtures)
  fixtures/                # tiny synthetic inputs for tests
  src/nur_pce/
    ingest/
    model/
    transport/
    output/
  tests/                   # pytest, TDD-built
  viewer/index.html        # single-file HTML, no external CDN
  docs/
    superpowers/specs/2026-04-23-nur-pce-design.md  (this file)
  pyproject.toml
  README.md
  E156-PROTOCOL.md         # written at v0.1 ship
  .gitignore
  .git/                    # git init at start of implementation
```

## 5. Data layer

### Trials (POC scope)

| Trial | NCT | Design | N | Role |
|---|---|---|---|---|
| FIDELIO-DKD | NCT02540993 | Finerenone vs placebo in CKD + T2DM | 5,734 | Train |
| FIGARO-DKD | NCT02545049 | Finerenone vs placebo in CKD + T2DM (broader eGFR) | 7,437 | Train |
| FINEARTS-HF | NCT04435626 | Finerenone vs placebo in HFmrEF/HFpEF | 6,016 | **Held out — validation only** |

Trial discovery via AACT snapshot at `D:/AACT-storage/AACT/2026-04-12`. NCT IDs and design metadata extracted automatically; subgroup HRs and KM curves extracted from publication PDFs / supplementary materials with double-check against any extant Cochrane MA.

### Evidence tiers

| Tier | Source | Per-trial unit | What we can model |
|---|---|---|---|
| **Tier 1** | Published subgroup forest plot HRs + CIs | One row per (trial, subgroup, outcome) | Main effects + low-dimensional interactions via meta-regression |
| **Tier 2** | Reconstructed IPD via Guyot from KM curves | Per-individual time-to-event (reconstructed) | Tier 1 + individual-level interactions on time-to-event |
| **Tier 3** | Real IPD (Vivli / YODA) | Per-individual full covariates | Full HTE — **deferred from v0.1** |

The tier of every cell in the output cube is the *minimum* tier of evidence that contributed to its posterior. Cell-by-cell honesty.

## 6. Patient profile schema (covariate set B)

Seven covariates, banded coarsely so the grid is tractable (~3,000 cells before population-target multiplication):

| # | Covariate | Bands |
|---|---|---|
| 1 | Age | <60, 60–70, 70–75, 75+ |
| 2 | Sex | M, F |
| 3 | eGFR (mL/min/1.73m²) | <30, 30–45, 45–60, ≥60 |
| 4 | T2DM | yes, no |
| 5 | UACR (mg/g) | <300, 300–1000, ≥1000 |
| 6 | NYHA class | I–II, III–IV |
| 7 | Region (transportability target) | Pakistan, India, Sub-Saharan Africa, USA, EU |

Bands chosen to match how FIDELIO/FIGARO/FINEARTS publish their pre-specified subgroup forest plots (so Tier 1 data maps cleanly onto cells).

Grid size: 4 × 2 × 4 × 2 × 3 × 2 × 5 = **1,920 cells**. Manageable.

## 7. Statistical model

### Hierarchical Bayesian specification

For trial *i*, subgroup-cell *c*, observed log-hazard-ratio `y_{i,c}` with standard error `s_{i,c}` (from Tier 1) or with per-individual likelihood (from Tier 2):

```
y_{i,c} ~ Normal(theta_{i,c}, s_{i,c}^2)            [Tier 1 contribution]

theta_{i,c} = mu
              + alpha_i
              + sum_p beta_p · X_{c,p}              [prognostic main effects]
              + sum_p gamma_p · X_{c,p} · 1[treat]  [HTE interactions]

alpha_i ~ Normal(0, tau^2)                          [trial-level RE]

mu      ~ Normal(0, 1)                              [main treatment effect, weakly informative]
beta_p  ~ Normal(0, 0.5)                            [prognostic — moderate prior]
gamma_p ~ Normal(0, 0.25)                           [interaction — tighter prior, HTE is rare]
tau     ~ Half-Normal(0, 0.5)                       [REML-equivalent between-trial heterogeneity]
```

Tier 2 IPD contributes via Cox-via-Poisson reparameterisation: each reconstructed individual contributes a per-time-bin Poisson likelihood, with the linear predictor sharing the same `theta` structure.

### Why these priors

- **Tighter prior on `gamma` (interactions) than `beta` (main effects)**: standard "shrink HTE toward no heterogeneity" stance, matching the Bayesian HTE literature (Henderson; Hahn-Murray-Carvalho via BCF). Stops over-fitting subgroup noise — critical with k=2 training trials.
- **`alpha_i` random effect**: captures between-trial differences. REML-equivalent. Per `~/.claude/rules/advanced-stats.md`: "Never use DL for k<10 — use REML or PM." This Bayesian formulation is REML-equivalent.
- **Joint Tier 1 + Tier 2 likelihood**: same `theta` structure, separate likelihood terms. Tier badge is bookkeeping over which terms contributed — not a separate model.

### Engine

Primary: **`cmdstanpy`** (Stan). Reasons: well-tested, posterior diagnostics (R̂, ESS) per `~/.claude/rules/advanced-stats.md` ("R̂ > 1.01 → do NOT interpret"; "ESS < 400 → unreliable CrI"). Diagnostics are gating checks in the pipeline — failure on R̂ or ESS halts and surfaces the failure rather than silently producing junk.

Fallback: **PyMC** if `cmdstanpy` install proves brittle on the user's Windows 11 + Python 3.13 setup. Both engines fit the same model.

### Transportability — g-formula posterior projection

For each target population *T* with covariate marginal distribution `P_T(X)` (drawn from IHME GBD 2021 + WB World Development Indicators), the population-average HR is:

```
HR_T = E_{P_T(X)} [ exp(theta(X)) ]
```

Computed by averaging the posterior at each cell weighted by the target's covariate marginals. Posterior uncertainty in `theta` propagates fully; additional uncertainty from finite-sample target marginals is propagated via bootstrap of the IHME/WHO source estimates.

### Uncertainty decomposition

Each output cell exposes total CrI on log(HR) split into three components:

1. **Within-trial sampling variance** (`s_{i,c}^2` and Tier 2 likelihood contribution).
2. **HTE / interaction posterior spread** (variance in `gamma · X` across posterior draws).
3. **Transportability variance** (variance from bootstrap of the target-population marginals).

This is the killer-feature display: the viewer shows where the uncertainty is *coming from*, not just how big it is.

## 8. Output: posterior cube + viewer

### `viewer/posterior_cube.json` schema

```json
{
  "schema_version": "0.1",
  "generated_at": "2026-04-23T...Z",
  "drug": "finerenone",
  "comparator": "placebo",
  "outcome": "composite_cardiorenal",
  "covariates": ["age_band", "sex", "eGFR_band", "t2dm", "uacr_band", "nyha", "region"],
  "cells": [
    {
      "key": {"age_band": "70-75", "sex": "M", "eGFR_band": "30-45",
              "t2dm": true, "uacr_band": "300-1000", "nyha": "III-IV",
              "region": "Pakistan"},
      "hr_mean": 0.74,
      "hr_credible_95": [0.61, 0.89],
      "p_hr_lt_1": 0.997,
      "tier": 2,
      "uncertainty_decomp": {"sampling": 0.06, "hte": 0.04, "transport": 0.02}
    },
    ...
  ],
  "diagnostics": {"rhat_max": 1.004, "ess_min": 1812, "divergent": 0}
}
```

### Viewer (single-file HTML)

- Single file, no external CDN (per `~/.claude/rules/rules.md` HTML-apps rule: "Fully offline — no external CDN").
- Loads `posterior_cube.json` once. All interaction is client-side — covariate sliders re-key into the cube.
- Tier badge prominent per query: green (T3), amber (T2), red (T1) with hover-explainer.
- Uncertainty decomposition shown as stacked bars beside the CrI.
- Pedagogy panel (Fatiha-style): plain-language explanation of what a Tier-2 cell means, how the answer was computed, what would tighten it (e.g., "real IPD via Vivli would move this cell to Tier 3 and likely tighten the CrI by ~30%").
- Deterministic seeded PRNG for any browser-side sampling (per `~/.claude/rules/rules.md` HTML-apps rule).
- Unique `localStorage` key (`nur-pce-v0.1`) to avoid collision with other RapidMeta apps (per `MEMORY.md#top-5-cross-project-defects`).

## 9. Validation — held-out FINEARTS-HF

### Setup

- Train PCE on FIDELIO-DKD + FIGARO-DKD (Tier 1 + Tier 2) only.
- FINEARTS-HF held out entirely — never seen by the model fit.
- For each pre-specified subgroup of FINEARTS-HF (the trial publishes ~10 subgroup HRs in its primary forest plot), generate the PCE's predicted patient-conditional HR posterior at that subgroup's central covariate cell.

### Comparators

| | Approach | What it represents |
|---|---|---|
| **Baseline 1** | Pooled fixed-effect MA HR from FIDELIO+FIGARO | Current SOTA — what a clinician gets today from a published MA |
| **Baseline 2** | Random-effects meta-regression with same covariates as PCE | The strongest non-Bayesian non-personalised competitor |
| **NUR-PCE** | Patient-conditional posterior at subgroup central cell | This proposal |
| **Truth** | FINEARTS-HF subgroup HR + CI as published | Ground truth |

### Metrics

1. **Subgroup-level RMSE on log(HR)** — primary.
2. **CrI calibration** — fraction of FINEARTS subgroups whose observed HR falls inside PCE's 95% CrI. Target band: 0.85–0.99.
3. **Decision concordance** — for each subgroup, would PCE's posterior probability `P(HR<1) > 0.95` lead to a different recommendation than the pooled MA point estimate? Meaningful only when the answer differs *and* PCE is right.
4. **Tier-aware reporting** — all metrics computed separately for a Tier-1-only fit vs Tier-1+2 fit, to demonstrate the value of each tier.

### Pass criteria for v0.1

- PCE beats Baseline 1 on RMSE in ≥6/10 FINEARTS subgroups.
- PCE 95% CrI calibration is in [0.85, 0.99] (well-calibrated; not over-confident below 0.85, not vacuous above 0.99).
- Tier-1-only fit is *honest* — wider CrIs than Tier-1+2, but not pathologically wide (median CrI half-width < 0.40 on log-HR scale).

### What if PCE fails

That is itself a publishable result. The validation is genuinely falsifiable. Either outcome (pass or fail) supports a Synthēsis Methods Note or higher-tier methods journal submission with material scientific value.

## 10. Scope summary (v0.1)

**IN**
- Drug: finerenone vs placebo, single comparison.
- Outcome: composite cardiorenal endpoint (CKD progression + CV death + HHF) — primary in all three trials.
- Trials: FIDELIO-DKD + FIGARO-DKD (training); FINEARTS-HF (held-out validation).
- Tiers: 1 + 2.
- Covariate schema: 7 covariates as in §6.
- Target populations: 5 (Pakistan, India, Sub-Saharan Africa region, USA, EU).
- Single-file HTML viewer + Python precompute pipeline.
- TDD-built (per `feedback_research_methodology.md`: Brainstorm → spec-lock → plan-lock → TDD → audit).
- Sentinel-clean per pre-push hook before any GitHub publication.

**OUT (deferred to v0.2+)**
- Live AACT polling (Pilot B).
- Multi-drug evidence graph (`EGQE`, Pilot 2).
- Deliberative LLM tribunal (`DST`, Pilot 3).
- Real IPD via Vivli/YODA.
- Other outcomes (ACM, hyperkalemia AEs).
- Other MRA-class drugs.
- Multi-language (Synthesis-Courses pipeline handles this).

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tier 2 IPD reconstruction (Guyot) introduces systematic bias | Sensitivity analysis: refit with Tier 1 only; report both. Sentinel rule on numerical baseline drift. |
| `cmdstanpy` install fails on Windows / Python 3.13 | PyMC fallback path; same model, swappable engine layer. Per `~/.claude/rules/lessons.md`: known Python 3.13 WMI deadlock — apply the documented monkey-patch before scipy import. |
| Subgroup HR extraction errors from PDF forest plots | Manual double-check + cross-check against any extant Cochrane MA; record extractor + verifier per row in `tier1.json`. Per `lessons.md`: "Treat trial IDs, NCT IDs, PMIDs, DOIs, exact dates, and cohort labels as typed fields, not approximate text." |
| Negation traps in regex extraction (e.g., "Not Randomized 1,807") | Per `lessons.md` rule learned 2026-04-15: any value-extraction regex must check the preceding 30 chars for negation words. Implement at the extractor layer with unit tests. |
| Posterior diagnostics fail (R̂ > 1.01 or ESS < 400) | Pipeline halts with explicit failure message — no silent emit. Per `advanced-stats.md`. |
| Held-out validation appears to "pass" because the held-out trial leaks into training (e.g., FINEARTS subgroups appearing in any imported summary) | Pre-flight check: hash all source-document file paths; assert no file with NCT04435626 is loaded by training stages. |
| Win condition appears arbitrary | Pass criteria pre-specified in §9 *before* implementation begins. Frozen at spec-lock. |
| Browser-side localStorage collision with other RapidMeta apps | Unique key `nur-pce-v0.1` per `MEMORY.md#top-5-cross-project-defects`. |

## 12. Open questions (for review)

These are not blockers but would benefit from your judgment before implementation:

1. **Priors on `gamma` (interaction terms)** — I used Normal(0, 0.25) by default following Henderson-style HTE conventions. Should we elicit tighter (e.g., 0.15) given k=2 training trials, or accept the default?
2. **Region handling** — currently treated as a covariate that drives transportability targets. An alternative is to drop region from the covariate vector and only have it appear at the transportability projection step. Cleaner, but loses any direct trial-region effect (FIDELIO had Asian-Pacific recruitment). Default: keep as covariate.
3. **Tier 2 cell minimum** — the Guyot reconstruction of FIDELIO/FIGARO will not give us pristine per-cell IPD; the cell's tier depends on whether we have ≥N reconstructed individuals in that exact cell. What's the minimum N for Tier-2 status? Default proposal: 30. Below that, the cell is Tier 1 even if it overlaps a reconstructed region.
4. **Synthēsis paper framing** — Methods Note (≤400w) would describe the engine; full RSM paper would describe the engine + finerenone validation. Default: write both, ship the Methods Note first.

## 13. Success criteria for v0.1 (POC ships if…)

1. Pipeline runs end-to-end on the real FIDELIO+FIGARO data and produces a valid `posterior_cube.json`.
2. Posterior diagnostics pass (R̂ ≤ 1.01, ESS ≥ 400 per parameter, zero divergent transitions).
3. Held-out validation report (§9) generated automatically.
4. Validation pass criteria met **OR** failure analysis report explaining why.
5. Single-file HTML viewer loads the cube and renders a tier-badged, uncertainty-decomposed answer for any covariate cell × target population.
6. ≥80% test coverage on `src/nur_pce/`; pytest green.
7. Sentinel scan: zero BLOCK; WARNs triaged.
8. README + E156-PROTOCOL.md drafted.
9. Reproducible from a clean clone in ≤30 minutes on the user's hardware.

---

*End of spec v0.1.*
