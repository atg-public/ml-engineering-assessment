import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_frame():
    """Two short event/tier series, already joined, as `build_daily_frame` returns them."""
    dates = pd.date_range("2026-08-01", periods=10, freq="D")
    rows = []
    for event_id, tier, base in [("201", "A", 100.0), ("202", "C", 30.0)]:
        for i, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "event_id": event_id,
                    "tier": tier,
                    "tickets_sold": float(i + 1),
                    "price": base,
                    "event_name": f"Show {event_id}",
                    "event_date": pd.Timestamp("2026-09-01"),
                    "capacity": 500,
                    "base_price": base,
                    "sessions": float(100 + 10 * i),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def training_frame():
    """A few hundred rows of noisy but learnable demand for smoke-testing the model."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-06-01", periods=60, freq="D")
    rows = []
    for e in range(4):
        for tier in "ABC":
            level = rng.uniform(3, 12)
            for day in dates:
                sessions = level * 40 * (1.3 if day.dayofweek >= 5 else 1.0)
                rows.append(
                    {
                        "date": day,
                        "event_id": str(300 + e),
                        "tier": tier,
                        "tickets_sold": float(rng.poisson(level)),
                        "price": 50.0,
                        "event_date": pd.Timestamp("2026-10-01"),
                        "capacity": 2000,
                        "base_price": 50.0,
                        "sessions": sessions,
                    }
                )
    return pd.DataFrame(rows)
