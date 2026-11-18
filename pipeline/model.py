"""Train and apply the daily demand model."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from pipeline.config import SEED, TARGET
from pipeline.features import FEATURES, training_rows

log = logging.getLogger(__name__)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: total absolute error over total volume."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / y_pred.sum())


def train_model(features: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, dict]:
    data = training_rows(features)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=300,
        learning_rate=0.05,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    metrics = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_wape": wape(y_val, model.predict(X_val)),
    }
    log.info("trained on %d rows, validation WAPE %.3f", metrics["n_train"], metrics["val_wape"])
    return model, metrics


def predict_demand(model: HistGradientBoostingRegressor, scoring: pd.DataFrame) -> pd.DataFrame:
    out = scoring.copy()
    out["forecast_daily_demand"] = np.clip(model.predict(scoring[FEATURES]), 0, None)
    return out
