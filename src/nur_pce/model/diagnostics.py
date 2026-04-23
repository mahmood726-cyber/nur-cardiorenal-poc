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
