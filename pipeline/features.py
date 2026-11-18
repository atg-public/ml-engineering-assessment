"""Feature engineering for the daily demand model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import TARGET, TIER_CODES

SERIES_KEYS = ["event_id", "tier"]

FEATURES = [
    "dow",
    "is_weekend",
    "days_to_event",
    "tier_code",
    "price_index",
    "sales_lag_1",
    "sales_lag_7",
    "sales_roll_7",
    "sessions_lag_1",
    "sessions_roll_7",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["days_to_event"] = (df["event_date"] - df["date"]).dt.days
    df["tier_code"] = df["tier"].map(TIER_CODES)
    df["price_index"] = df["price"] / df["base_price"]
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(SERIES_KEYS + ["date"]).copy()
    by_series = df.groupby(SERIES_KEYS, sort=False)

    df["sales_lag_1"] = by_series[TARGET].shift(1)
    df["sales_lag_7"] = by_series[TARGET].shift(7)
    df["sales_roll_7"] = by_series[TARGET].transform(lambda s: s.rolling(7, min_periods=1).mean())

    df["sessions_lag_1"] = by_series["sessions"].shift(1)
    df["sessions_roll_7"] = by_series["sessions"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_history_features(add_calendar_features(df))


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known target and a full feature history."""
    return features.dropna(subset=FEATURES + [TARGET])


def make_scoring_frame(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per live event/tier for the day after `as_of`.

    History features for that day come from the observed days before it, so
    the target and same-day traffic are left empty.
    """
    latest = history[history["date"] == as_of]
    next_day = latest.copy()
    next_day["date"] = as_of + pd.Timedelta(days=1)
    next_day[TARGET] = np.nan
    next_day["sessions"] = np.nan
    next_day = next_day[next_day["event_date"] > next_day["date"]]

    frame = build_features(pd.concat([history, next_day], ignore_index=True))
    scoring = frame[frame["date"] == as_of + pd.Timedelta(days=1)]
    return scoring.dropna(subset=FEATURES).reset_index(drop=True)
