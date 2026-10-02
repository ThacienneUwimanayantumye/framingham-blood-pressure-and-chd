from __future__ import annotations

import statsmodels.api as sm
import statsmodels.formula.api as smf

OLS_FORMULAS = {
    "sbp_base": "SBP ~ sFRW + SEX + sCHOL",
    "sbp_age": "SBP ~ sFRW + SEX + sCHOL + sAGE",
    "sbp_interactions": (
        "SBP ~ sFRW + SEX + sCHOL + sAGE + sFRW:SEX + sFRW:sCHOL + "
        "sFRW:sAGE + SEX:sCHOL + SEX:sAGE + sCHOL:sAGE"
    ),
    "sbp_smoking": (
        "SBP ~ sFRW + SEX + sCHOL + sAGE + sCIG + "
        "sFRW:SEX + sFRW:sCHOL + sFRW:sAGE + SEX:sCHOL + SEX:sAGE + "
        "sCHOL:sAGE + sFRW:sCIG + SEX:sCIG + sCHOL:sCIG + sAGE:sCIG"
    ),
}

HBP_FORMULAS = {
    "hbp_weight_sex": "HIGH_BP ~ sFRW + SEX + SEX:sFRW",
    "hbp_age": "HIGH_BP ~ sFRW + SEX + SEX:sFRW + sAGE + sAGE:sFRW + sAGE:SEX",
}

# Nested CHD specifications. Headline model is main_effects.
CHD_FORMULAS = {
    "without_age_sex": (
        "incident_chd ~ sCHOL + sCIG + sFRW + sCHOL:sCIG + sCHOL:sFRW + sCIG:sFRW"
    ),
    "main_effects": "incident_chd ~ sCHOL + sCIG + sFRW + sAGE + SEX",
    "main_plus_interactions": (
        "incident_chd ~ sCHOL + sCIG + sFRW + sAGE + SEX + "
        "sCHOL:sCIG + sCHOL:sFRW + sCIG:sFRW"
    ),
}

CHD_FORMULA = CHD_FORMULAS["main_effects"]
HBP_FORMULA = HBP_FORMULAS["hbp_age"]
OLS_FORMULA = OLS_FORMULAS["sbp_age"]

SCALE_PREDICTORS = ("AGE", "FRW", "CHOL", "CIG")


def fit_ols(df, formula: str):
    return smf.ols(formula, data=df).fit()


def fit_logit(df, formula: str):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
