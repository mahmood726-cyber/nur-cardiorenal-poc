from __future__ import annotations
import json
from pathlib import Path
import shutil
import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    drv = webdriver.Chrome(options=opts)
    yield drv
    drv.quit()


def test_viewer_loads_cube_and_renders_tier_badge(driver, tmp_path_factory):
    viewer_src = Path(__file__).parent.parent / "viewer" / "index.html"
    cube_src = Path(__file__).parent.parent / "fixtures" / "cube_synth.json"
    if not cube_src.exists():
        cube_src = tmp_path_factory.mktemp("c") / "cube_synth.json"
        cube_src.write_text(json.dumps({
            "schema_version": "0.1",
            "generated_at": "2026-04-23T00:00:00Z",
            "drug": "finerenone", "comparator": "placebo",
            "outcome": "composite_cardiorenal",
            "covariates": ["age_band", "sex", "eGFR_band", "t2dm",
                           "uacr_band", "nyha", "region"],
            "cells": [{
                "key": {"age_band": "60-70", "sex": "M", "eGFR_band": "30-45",
                        "t2dm": True, "uacr_band": "300-1000",
                        "nyha": "I-II", "region": "USA"},
                "hr_mean": 0.74, "hr_credible_95": [0.61, 0.89],
                "p_hr_lt_1": 0.99, "tier": 2,
                "uncertainty_decomp": {"sampling": 0.06, "hte": 0.04, "transport": 0.02},
            }],
            "diagnostics": {"rhat_max": 1.004, "ess_min": 1812, "divergent": 0},
        }))

    work = tmp_path_factory.mktemp("viewer")
    shutil.copy(viewer_src, work / "index.html")
    shutil.copy(cube_src, work / "posterior_cube.json")
    driver.get((work / "index.html").as_uri())
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "tier-badge"))
    )
    badge = driver.find_element(By.ID, "tier-badge").text
    assert badge.lower().startswith("tier")
