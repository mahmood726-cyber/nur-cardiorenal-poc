from __future__ import annotations
import pytest
from pydantic import ValidationError
from nur_pce.schema import (
    Trial, CovariateKey, Tier1Row, Tier2Record, CubeCell, UncertaintyDecomp,
    AGE_BANDS, EGFR_BANDS, UACR_BANDS, NYHA_CLASSES, REGIONS,
)


def test_covariate_key_validates_bands():
    key = CovariateKey(
        age_band="60-70", sex="M", eGFR_band="30-45",
        t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
    )
    assert key.age_band == "60-70"


def test_covariate_key_rejects_bad_age_band():
    with pytest.raises(ValidationError):
        CovariateKey(
            age_band="999", sex="M", eGFR_band="30-45",
            t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
        )


def test_tier1_row_requires_extractor_and_verifier():
    with pytest.raises(ValidationError):
        Tier1Row(
            trial_id="NCT02540993",
            subgroup_key=CovariateKey(
                age_band="60-70", sex="M", eGFR_band="30-45",
                t2dm=True, uacr_band="300-1000", nyha="I-II", region="USA",
            ),
            log_hr=-0.30, se=0.12, outcome="composite_cardiorenal",
        )  # missing extractor + verifier


def test_cube_cell_uncertainty_decomp_sums_to_total():
    cell = CubeCell(
        key=CovariateKey(
            age_band="70-75", sex="M", eGFR_band="30-45",
            t2dm=True, uacr_band="300-1000", nyha="III-IV", region="Pakistan",
        ),
        hr_mean=0.74, hr_credible_95=(0.61, 0.89), p_hr_lt_1=0.997,
        tier=2,
        uncertainty_decomp=UncertaintyDecomp(sampling=0.06, hte=0.04, transport=0.02),
    )
    assert cell.uncertainty_decomp.total() == pytest.approx(0.12)


def test_constants_match_spec():
    assert AGE_BANDS == ("<60", "60-70", "70-75", "75+")
    assert EGFR_BANDS == ("<30", "30-45", "45-60", ">=60")
    assert UACR_BANDS == ("<300", "300-1000", ">=1000")
    assert NYHA_CLASSES == ("I-II", "III-IV")
    assert REGIONS == ("Pakistan", "India", "Sub-Saharan Africa", "USA", "EU")
