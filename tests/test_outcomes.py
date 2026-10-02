import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framreg.data import add_outcomes, incident_sample, load_fram
from framreg.scale import rescale


def pd_series():
    return pd.Series([10.0, 20.0, 30.0, 40.0])


def test_rescale_mean_near_zero_and_sd_half():
    z = rescale(pd_series())
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=1) - 0.5) < 1e-12


def test_high_bp_definition():
    df = add_outcomes(load_fram())
    expected = ((df["SBP"] >= 140) | (df["DBP"] >= 90)).astype(int)
    assert df["HIGH_BP"].equals(expected)


def test_prevalent_chd_is_yrs_pre():
    df = add_outcomes(load_fram())
    pre = df["YRS_CHD"].astype(str).str.strip().str.lower() == "pre"
    assert int(df["prevalent_chd"].sum()) == int(pre.sum())
    assert int(df["prevalent_chd"].sum()) == 42
    assert (df.loc[pre, "CHD"] > 0).all()


def test_incident_chd_excludes_prevalent():
    df = add_outcomes(load_fram())
    assert int((df["incident_chd"] & df["prevalent_chd"]).sum()) == 0
    assert int(df["incident_chd"].sum() + df["prevalent_chd"].sum()) == int(df["hasCHD"].sum())
    inc = incident_sample(df)
    assert inc["prevalent_chd"].sum() == 0
    assert len(inc) == len(df) - 42


def test_incident_majority_is_no_event():
    df = incident_sample(add_outcomes(load_fram()))
    p = df["incident_chd"].mean()
    assert p < 0.5
    assert abs(max(p, 1 - p) - (1 - p)) < 1e-12
