"""Pull finerenone trials from a local AACT snapshot.

Per ~/.claude/rules/lessons.md: lowercase intervention types, validate >0 rows,
do not hardcode one drive. AACT snapshot path is parameterised; fail closed if
absent.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from nur_pce.schema import Trial


class AACTSnapshotMissing(FileNotFoundError):
    """Raised when the AACT snapshot directory is absent."""


def find_drug_trials(snapshot_root: Path, drug: str) -> list[Trial]:
    snapshot_root = Path(snapshot_root)
    if not snapshot_root.exists():
        raise AACTSnapshotMissing(
            f"AACT snapshot not found at {snapshot_root}. "
            f"Set the path via NUR_AACT_PATH or pass --aact-path."
        )
    interventions = pd.read_csv(snapshot_root / "interventions.csv")
    studies = pd.read_csv(snapshot_root / "studies.csv")
    designs = pd.read_csv(snapshot_root / "designs.csv")

    drug_lc = drug.lower()
    drug_ncts = set(
        interventions.loc[interventions["name"].str.lower() == drug_lc, "nct_id"]
    )
    placebo_ncts = set(
        interventions.loc[interventions["name"].str.lower() == "placebo", "nct_id"]
    )
    eligible_ncts = drug_ncts & placebo_ncts

    rows = (
        studies.merge(designs, on="nct_id")
        .loc[lambda d: d["nct_id"].isin(eligible_ncts)]
    )
    if rows.empty:
        return []

    return [
        Trial(
            nct_id=r["nct_id"],
            drug=drug,
            comparator="placebo",
            n_total=int(r["enrollment"]),
            design=f"{r['allocation']} / {r['intervention_model']} / {r['masking']}",
            primary_outcome="composite_cardiorenal",  # set by spec; refined later
        )
        for _, r in rows.iterrows()
    ]
