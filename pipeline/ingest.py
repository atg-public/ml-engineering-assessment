"""Load the raw extracts and join them into one row per event, tier and day."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_events(data_dir: Path) -> pd.DataFrame:
    """Event/tier reference data: capacity, base price and show date."""
    events = pd.read_csv(data_dir / "events.csv", parse_dates=["event_date"])
    return events


def load_sales(data_dir: Path) -> pd.DataFrame:
    """Daily ticket sales per event and tier.

    The box office can emit more than one row for a day (e.g. late
    adjustments), so collapse to a single daily total.
    """
    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["date"])
    daily = sales.groupby(["date", "event_id", "tier"], as_index=False).agg(
        tickets_sold=("tickets_sold", "sum"),
        price=("price", "last"),
    )
    return daily


def load_traffic(data_dir: Path) -> pd.DataFrame:
    """Daily web sessions per event, from the analytics export."""
    traffic = pd.read_csv(data_dir / "web_traffic.csv", parse_dates=["date"], dtype={"event_id": str})
    return traffic


def build_daily_frame(sales: pd.DataFrame, traffic: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event reference data and web traffic to daily sales."""
    df = sales.merge(events, on=["event_id", "tier"], how="left", validate="many_to_one")

    # Analytics IDs are strings, so align before joining.
    df["event_id"] = df["event_id"].astype(str)
    df = df.merge(traffic[["date", "event_id", "sessions"]], on=["date", "event_id"], how="left")

    return df.sort_values(["event_id", "tier", "date"]).reset_index(drop=True)


def load_all(data_dir: Path, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    events = load_events(data_dir)
    sales = load_sales(data_dir)
    traffic = load_traffic(data_dir)
    if as_of is not None:
        sales = sales[sales["date"] <= as_of]
        traffic = traffic[traffic["date"] <= as_of]

    df = build_daily_frame(sales, traffic, events)
    log.info(
        "loaded %d sales rows for %d events, %s to %s",
        len(df),
        df["event_id"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df
