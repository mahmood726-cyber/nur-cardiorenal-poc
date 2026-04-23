"""Pytest fixtures shared across the suite."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src/ is importable
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"
