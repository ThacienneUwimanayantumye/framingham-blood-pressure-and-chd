from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import CHD_FORMULA, HBP_FORMULA, OLS_FORMULA, fit_logit, fit_ols


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def plot_sbp_by_weight_age(df: pd.DataFrame, path: Path) -> None:
    _style()
    fit = fit_ols(df, OLS_FORMULA)
    grid = np.linspace(df["sFRW"].min(), df["sFRW"].max(), 80)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    women = df[df["SEX"] == "female"]
    ax.scatter(women["sFRW"], women["SBP"], s=12, alpha=0.35, color="#5b6b75", label="Women")
    for age, colour, label in [
        (-1, "#215C8C", "Younger (sAGE = −1)"),
        (0, "#C05621", "Mean age"),
        (1, "#7B2D26", "Older (sAGE = +1)"),
    ]:
        pred = pd.DataFrame(
            {"sFRW": grid, "SEX": "female", "sCHOL": 0.0, "sAGE": age}
        )
        ax.plot(grid, fit.predict(pred), color=colour, lw=2, label=label)
    ax.set_xlabel("Relative weight (Gelman-scaled)")
    ax.set_ylabel("Systolic BP (mmHg)")
    ax.set_title("Women: systolic BP vs weight, at three ages")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_hbp_probability(df: pd.DataFrame, path: Path) -> None:
    _style()
    fit = fit_logit(df, HBP_FORMULA)
    grid = np.linspace(df["sFRW"].min(), df["sFRW"].max(), 80)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, sex in zip(axes, ("female", "male")):
        subset = df[df["SEX"] == sex]
        ax.scatter(subset["sFRW"], subset["HIGH_BP"], s=10, alpha=0.25, color="#24323d")
        for age, colour, label in [
            (-1, "#215C8C", "Younger"),
            (0, "#5b6b75", "Mean age"),
            (1, "#C05621", "Older"),
        ]:
            pred = pd.DataFrame({"sFRW": grid, "SEX": sex, "sAGE": age})
            ax.plot(grid, fit.predict(pred), color=colour, lw=2, label=label)
        ax.set_title(sex.capitalize())
        ax.set_xlabel("Relative weight (Gelman-scaled)")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("P(high blood pressure)")
    axes[1].legend(frameon=False, title="Age")
    fig.suptitle("Predicted high BP vs weight, by sex and age", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_chd_by_smoking(df: pd.DataFrame, path: Path) -> None:
    _style()
    fit = fit_logit(df, CHD_FORMULA)
    grid = np.linspace(df["sCIG"].min(), df["sCIG"].max(), 80)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        df["sCIG"],
        df["incident_chd"],
        s=12,
        alpha=0.3,
        color="#24323d",
        label="Observed",
    )
    for sex, colour, label in [
        ("female", "#215C8C", "Women, mean age"),
        ("male", "#C05621", "Men, mean age"),
    ]:
        pred = pd.DataFrame(
            {
                "sCIG": grid,
                "sCHOL": 0.0,
                "sFRW": 0.0,
                "sAGE": 0.0,
                "SEX": sex,
            }
        )
        ax.plot(grid, fit.predict(pred), color=colour, lw=2.2, label=label)
    ax.set_xlabel("Cigarettes (Gelman-scaled)")
    ax.set_ylabel("P(later CHD record)")
    ax.set_title("Incident CHD vs smoking, cholesterol and weight at their means")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
