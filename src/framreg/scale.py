"""Gelman-style rescale: (x - mean) / (2 * sd)."""

from __future__ import annotations

import pandas as pd


def rescale(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (2.0 * s.std(ddof=1))


CONTINUOUS = ("AGE", "FRW", "SBP", "DBP", "CHOL", "CIG")


def add_scaled_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in CONTINUOUS:
        out[f"s{name}"] = rescale(out[name])
    return out


def scale_raw_point(df: pd.DataFrame, raw: dict[str, float]) -> dict[str, float]:
    """Scale a new person's raw values using the sample mean and 2*sd."""
    point = {}
    for name, value in raw.items():
        point[f"s{name}"] = (value - df[name].mean()) / (2.0 * df[name].std(ddof=1))
    return point
