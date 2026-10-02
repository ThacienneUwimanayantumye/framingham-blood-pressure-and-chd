from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from .models import SCALE_PREDICTORS, fit_logit


def majority_accuracy(y: pd.Series) -> float:
    p = float(y.mean())
    return max(p, 1.0 - p)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    pos = tp + fn
    neg = tn + fp
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "prevalence": float(y_true.mean()),
        "majority_accuracy": float(max(y_true.mean(), 1.0 - y_true.mean())),
        "sensitivity": float(tp / pos) if pos else 0.0,
        "specificity": float(tn / neg) if neg else 0.0,
        "pred_positive_rate": float(pred.mean()),
        "mean_predicted": float(proba.mean()),
    }


def scale_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = SCALE_PREDICTORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gelman-scale predictors from the training fold only."""
    train = train.copy()
    test = test.copy()
    for name in cols:
        mu = train[name].mean()
        sd = train[name].std(ddof=1)
        train[f"s{name}"] = (train[name] - mu) / (2.0 * sd)
        test[f"s{name}"] = (test[name] - mu) / (2.0 * sd)
    return train, test


def monte_carlo_cv(
    df: pd.DataFrame,
    formula: str,
    y_col: str,
    n_splits: int = 100,
    train_size: float = 0.8,
    seed: int = 1,
    scale_in_fold: bool = True,
) -> pd.DataFrame:
    """Repeated stratified 80/20 splits.

    Predictor scaling is refit on each training fold unless scale_in_fold is
    False (in which case sAGE/sFRW/… must already exist on df).
    """
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=train_size, random_state=seed
    )
    rows = []
    y = df[y_col]
    for i, (train_idx, test_idx) in enumerate(splitter.split(df, y)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if scale_in_fold:
            train, test = scale_fold(train, test)
        model = fit_logit(train, formula)
        proba = model.predict(test)
        metrics = classification_metrics(test[y_col], proba)
        metrics["split"] = i
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> pd.Series:
    keep = [
        "accuracy",
        "roc_auc",
        "brier",
        "majority_accuracy",
        "sensitivity",
        "pred_positive_rate",
    ]
    return cv[keep].mean()
