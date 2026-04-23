# E156-PROTOCOL — NUR-PCE

**Project**: NUR-PCE — Personalised Counterfactual Engine (POC)
**Scope**: finerenone vs placebo, composite cardiorenal outcome.
**Started**: 2026-04-23
**Spec frozen**: 2026-04-23 (commit a6de22e)
**Plan frozen**: 2026-04-23 (commit af9ccc5; amendment 6264d29)
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
