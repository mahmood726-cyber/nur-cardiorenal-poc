# scripts/preflight.py
"""Preflight check for NUR-PCE implementation prerequisites.

Fails closed with a specific action list if any prereq is missing.
Run before starting any other task.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 11)
AACT_PATH = Path("D:/AACT-storage/AACT/2026-04-12")
IHME_PATH = Path("D:/Projects/ihme-data-lakehouse")
WHO_PATH = Path("D:/Projects/who-data-lakehouse")
WB_PATH = Path("D:/Projects/wb-data-lakehouse")


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
