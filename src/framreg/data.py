from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "fram.txt"


def load_fram(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else RAW_PATH
    return pd.read_table(path, sep="\t")


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Binary outcomes used in the public analysis.

    HIGH_BP follows the usual 140/90 mmHg cut-off on the exam measurements.

    hasCHD is CHD > 0. It mixes people who already had CHD at the exam
    (YRS_CHD == "pre") with later records.

    incident_chd is a later CHD record among people who were free of CHD at
    the exam. Prevalent cases are flagged separately and dropped from the
    CHD models.
    """
    out = df.copy()
    out["HIGH_BP"] = ((out["SBP"] >= 140) | (out["DBP"] >= 90)).astype(int)
    out["hasCHD"] = (out["CHD"] > 0).astype(int)
    yrs = out["YRS_CHD"].astype(str).str.strip().str.lower()
    out["prevalent_chd"] = (yrs == "pre").astype(int)
    out["incident_chd"] = ((out["hasCHD"] == 1) & (out["prevalent_chd"] == 0)).astype(int)
    return out


def incident_sample(df: pd.DataFrame) -> pd.DataFrame:
    """People without CHD at the exam (drop YRS_CHD == pre)."""
    if "prevalent_chd" not in df.columns:
        df = add_outcomes(df)
    return df.loc[df["prevalent_chd"] == 0].copy()
