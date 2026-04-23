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
