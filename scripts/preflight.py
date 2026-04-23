# scripts/preflight.py
"""Preflight check for NUR-PCE implementation prerequisites.

Fails closed with a specific action list if any prereq is missing.
Run before starting any other task.

Data paths are resolved from environment variables or candidate roots so
this script is not tied to any single developer machine (lessons.md#code-quality).

    AACT_PATH   — env var NUR_AACT_PATH, or auto-discover under C: / D:
    IHME_PATH   — env var NUR_IHME_PATH, or auto-discover under C: / D:
    WHO_PATH    — env var NUR_WHO_PATH,  or auto-discover under C: / D:
    WB_PATH     — env var NUR_WB_PATH,   or auto-discover under C: / D:
"""
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 11)

# ---------------------------------------------------------------------------
# Candidate-root discovery (lessons.md#CT.gov — "do not hardcode one drive").
# Set the env var to skip discovery entirely.
# ---------------------------------------------------------------------------
_CANDIDATE_ROOTS = [Path("C:/"), Path("D:/")]


def _resolve_path(env_var: str, *relative_candidates: str) -> Path | None:
    """Return path from env var if set, else first existing candidate, else None.

    Returns None when no candidate exists — `check_path` then reports the prereq
    as missing with an explicit env-var hint, instead of silently fabricating a
    plausible-looking path. Per code-review §critical-2 (lessons.md#code-quality).
    """
    if env_var in os.environ:
        return Path(os.environ[env_var])
    for root in _CANDIDATE_ROOTS:
        for rel in relative_candidates:
            p = root / rel
            if p.exists():
                return p
    return None


AACT_PATH = _resolve_path("NUR_AACT_PATH", "AACT-storage/AACT/2026-04-12")
IHME_PATH = _resolve_path("NUR_IHME_PATH", "Projects/ihme-data-lakehouse")
WHO_PATH = _resolve_path("NUR_WHO_PATH", "Projects/who-data-lakehouse")
WB_PATH = _resolve_path("NUR_WB_PATH", "Projects/wb-data-lakehouse")


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


def check_path(p: Path | None, label: str, env_var: str) -> tuple[bool, str]:
    if p is None:
        return False, f"{label} not found in any candidate root; set {env_var}"
    if not p.exists():
        return False, f"{label} path missing: {p}"
    return True, f"{label} OK at {p}"


def check_pymc() -> tuple[bool, str]:
    try:
        import pymc  # type: ignore
        return True, f"pymc {pymc.__version__} OK (Windows-friendly Bayesian engine; spec §11 fallback)"
    except ImportError:
        return False, "pymc NOT installed (pip install pymc)"


def main() -> int:
    checks = [
        ("python", check_python()),
        ("pydantic", check_module("pydantic")),
        ("pytest", check_module("pytest")),
        ("numpy", check_module("numpy")),
        ("scipy", check_module("scipy")),
        ("pandas", check_module("pandas")),
        ("pyarrow", check_module("pyarrow")),
        ("pymc", check_pymc()),
        ("aact", check_path(AACT_PATH, "AACT snapshot", "NUR_AACT_PATH")),
        ("ihme", check_path(IHME_PATH, "IHME lakehouse", "NUR_IHME_PATH")),
        ("who", check_path(WHO_PATH, "WHO lakehouse", "NUR_WHO_PATH")),
        ("wb", check_path(WB_PATH, "WB lakehouse", "NUR_WB_PATH")),
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
