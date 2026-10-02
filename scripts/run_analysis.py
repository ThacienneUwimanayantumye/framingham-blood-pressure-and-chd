#!/usr/bin/env python3
"""Fit the Framingham models, write nested comparisons, CV metrics, and figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from framreg.data import add_outcomes, incident_sample, load_fram
from framreg.evaluate import classification_metrics, monte_carlo_cv, summarise_cv
from framreg.models import (
    CHD_FORMULA,
    CHD_FORMULAS,
    HBP_FORMULA,
    HBP_FORMULAS,
    OLS_FORMULAS,
    fit_logit,
    fit_ols,
)
from framreg.plots import plot_chd_by_smoking, plot_hbp_probability, plot_sbp_by_weight_age
from framreg.scale import add_scaled_columns, scale_raw_point


def main() -> None:
    derived = ROOT / "data" / "derived"
    figures = ROOT / "docs" / "figures"
    derived.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    raw = add_outcomes(load_fram())
    df = add_scaled_columns(raw)
    chd_df = incident_sample(df)
    n = len(df)
    findings = {
        "n": n,
        "n_female": int((df["SEX"] == "female").sum()),
        "n_male": int((df["SEX"] == "male").sum()),
        "high_bp_rate": float(df["HIGH_BP"].mean()),
        "n_prevalent_chd": int(df["prevalent_chd"].sum()),
        "n_incident_eligible": int(len(chd_df)),
        "incident_chd_rate": float(chd_df["incident_chd"].mean()),
        "haschd_mixed_rate": float(df["hasCHD"].mean()),
    }

    ols_rows = []
    for name, formula in OLS_FORMULAS.items():
        fit = fit_ols(df, formula)
        ols_rows.append(
            {
                "model": name,
                "formula": formula,
                "r_squared": fit.rsquared,
                "adj_r_squared": fit.rsquared_adj,
                "aic": fit.aic,
                "n_params": int(fit.df_model) + 1,
            }
        )
    ols_table = pd.DataFrame(ols_rows)
    ols_table.to_csv(derived / "ols_sbp.csv", index=False)

    hbp_in_sample = []
    for name, formula in HBP_FORMULAS.items():
        fit = fit_logit(df, formula)
        metrics = classification_metrics(df["HIGH_BP"], fit.predict(df))
        metrics["model"] = name
        metrics["aic"] = float(fit.aic)
        hbp_in_sample.append(metrics)
    pd.DataFrame(hbp_in_sample).to_csv(derived / "hbp_in_sample.csv", index=False)

    chd_spec = []
    for name, formula in CHD_FORMULAS.items():
        fit = fit_logit(chd_df, formula)
        metrics = classification_metrics(chd_df["incident_chd"], fit.predict(chd_df))
        metrics["model"] = name
        metrics["formula"] = formula
        metrics["aic"] = float(fit.aic)
        chd_spec.append(metrics)
    pd.DataFrame(chd_spec).to_csv(derived / "chd_spec_comparison.csv", index=False)

    chd_fit = fit_logit(chd_df, CHD_FORMULA)
    coefs = pd.DataFrame(
        {
            "term": chd_fit.params.index,
            "coef": chd_fit.params.values,
            "std_err": chd_fit.bse.values,
            "p_value": chd_fit.pvalues.values,
        }
    )
    coefs.to_csv(derived / "chd_main_coefs.csv", index=False)

    # CV uses the unscaled table so each training fold sets its own mean and 2*sd.
    hbp_cv = monte_carlo_cv(raw, HBP_FORMULA, "HIGH_BP")
    chd_cv = monte_carlo_cv(incident_sample(raw), CHD_FORMULA, "incident_chd")
    hbp_cv.to_csv(derived / "hbp_cv_repeats.csv", index=False)
    chd_cv.to_csv(derived / "chd_cv_repeats.csv", index=False)

    hbp_mean = summarise_cv(hbp_cv)
    chd_mean = summarise_cv(chd_cv)
    summary = pd.DataFrame(
        [
            {"outcome": "HIGH_BP", **hbp_mean.to_dict()},
            {"outcome": "incident_chd", **chd_mean.to_dict()},
        ]
    )
    summary.to_csv(derived / "cv_summary.csv", index=False)

    point_raw = {"CHOL": 200.0, "CIG": 17.0, "FRW": 100.0, "AGE": float(df["AGE"].mean())}
    point = scale_raw_point(df, point_raw)
    example = {}
    for sex in ("female", "male"):
        pred_df = pd.DataFrame([{**point, "SEX": sex}])
        example[sex] = float(chd_fit.predict(pred_df).iloc[0])
    findings["example_person"] = {
        "raw": point_raw,
        "scaled": point,
        "p_incident_chd": example,
        "note": (
            "Cholesterol 200, 17 cigarettes/day, relative weight 100, sample-mean age. "
            "Scaled with (x − mean) / (2 × sd) on the raw columns."
        ),
    }
    findings["cv"] = {
        "HIGH_BP": hbp_mean.to_dict(),
        "incident_chd": chd_mean.to_dict(),
    }
    findings["chd_in_sample_auc"] = {
        row["model"]: float(row["roc_auc"]) for row in chd_spec
    }
    findings["chd_in_sample_aic"] = {
        row["model"]: float(row["aic"]) for row in chd_spec
    }
    findings["sbp_r_squared"] = {
        row["model"]: float(row["r_squared"]) for row in ols_rows
    }

    plot_sbp_by_weight_age(df, figures / "sbp_weight_age_women.png")
    plot_hbp_probability(df, figures / "hbp_weight_sex_age.png")
    plot_chd_by_smoking(chd_df, figures / "chd_smoking.png")

    (derived / "findings.json").write_text(json.dumps(findings, indent=2))

    print("n =", n, "incident-eligible =", len(chd_df))
    print(ols_table.to_string(index=False))
    print("\nCHD specification comparison (incident sample):")
    print(pd.DataFrame(chd_spec)[["model", "aic", "roc_auc", "accuracy", "majority_accuracy"]].to_string(index=False))
    print("\nHBP 100× stratified 80/20 CV (mean):")
    print(hbp_mean.to_string())
    print("\nIncident CHD 100× stratified 80/20 CV (mean):")
    print(chd_mean.to_string())
    print("\nExample P(incident CHD) woman={:.1%} man={:.1%}".format(example["female"], example["male"]))
    print("Wrote tables to", derived)
    print("Wrote figures to", figures)


if __name__ == "__main__":
    main()
