"""Pydantic data classes for NUR-PCE.

Constants are typed as Literal so Pydantic enforces band membership at parse time.
"""
from __future__ import annotations
from typing import Literal, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

AGE_BANDS = ("<60", "60-70", "70-75", "75+")
EGFR_BANDS = ("<30", "30-45", "45-60", ">=60")
UACR_BANDS = ("<300", "300-1000", ">=1000")
NYHA_CLASSES = ("I-II", "III-IV")
REGIONS = ("Pakistan", "India", "Sub-Saharan Africa", "USA", "EU")

AgeBand = Literal["<60", "60-70", "70-75", "75+"]
Sex = Literal["M", "F"]
EgfrBand = Literal["<30", "30-45", "45-60", ">=60"]
UacrBand = Literal["<300", "300-1000", ">=1000"]
NyhaClass = Literal["I-II", "III-IV"]
Region = Literal["Pakistan", "India", "Sub-Saharan Africa", "USA", "EU"]
Tier = Literal[1, 2, 3]


class CovariateKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    age_band: AgeBand
    sex: Sex
    eGFR_band: EgfrBand
    t2dm: bool
    uacr_band: UacrBand
    nyha: NyhaClass
    region: Region


class Trial(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    drug: str
    comparator: str
    n_total: int = Field(gt=0)
    design: str
    primary_outcome: str


class Tier1Row(BaseModel):
    trial_id: str = Field(pattern=r"^NCT\d{8}$")
    subgroup_key: CovariateKey
    log_hr: float
    se: float = Field(gt=0)
    outcome: str
    extractor: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    source_doc: str = Field(min_length=1)
    source_page: int | None = None


class Tier2Record(BaseModel):
    trial_id: str = Field(pattern=r"^NCT\d{8}$")
    subject_id: str
    arm: Literal["treatment", "control"]
    time: float = Field(ge=0)
    event: int = Field(ge=0, le=1)
    covariates: CovariateKey
    reconstructed: bool = True


class UncertaintyDecomp(BaseModel):
    sampling: float = Field(ge=0)
    hte: float = Field(ge=0)
    transport: float = Field(ge=0)

    def total(self) -> float:
        return self.sampling + self.hte + self.transport


class CubeCell(BaseModel):
    key: CovariateKey
    hr_mean: float = Field(gt=0)
    hr_credible_95: Tuple[float, float]
    p_hr_lt_1: float = Field(ge=0, le=1)
    tier: Tier
    uncertainty_decomp: UncertaintyDecomp

    @field_validator("hr_credible_95")
    @classmethod
    def _ci_ordered(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if not (lo > 0 and hi > 0 and lo <= hi):
            raise ValueError("hr_credible_95 must be (lo, hi) with 0 < lo <= hi")
        return v


class Cube(BaseModel):
    schema_version: str = "0.1"
    generated_at: str
    drug: str
    comparator: str
    outcome: str
    covariates: list[str]
    cells: list[CubeCell]
    diagnostics: dict[str, float]
