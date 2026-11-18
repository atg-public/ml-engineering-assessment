"""Turn demand forecasts into price recommendations."""
from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import PRICE_SENSITIVITY, TARGET, TARGET_SELL_THROUGH
from pipeline.inventory_client import InventoryAPIError, InventoryClient

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "as_of",
    "event_id",
    "event_name",
    "tier",
    "event_date",
    "current_price",
    "forecast_daily_demand",
    "remaining",
    "inventory_source",
    "projected_sell_through",
    "recommended_price",
    "price_change_pct",
]


def projected_sell_through(daily_demand: float, days_to_event: int, remaining: int) -> float:
    """Share of the remaining seats we expect to sell before the show at the current rate."""
    return daily_demand * days_to_event / max(remaining, 1)


def recommend_price(current_price: float, sell_through: float) -> float:
    """Raise price when selling ahead of target, lower it when behind."""
    change = PRICE_SENSITIVITY * (sell_through - TARGET_SELL_THROUGH)
    return round(current_price * (1 + change), 2)


def fallback_inventory(history: pd.DataFrame) -> pd.DataFrame:
    """Remaining seats and latest price derived from our own sales history."""
    latest = history.sort_values("date").groupby(["event_id", "tier"]).tail(1)
    sold = history.groupby(["event_id", "tier"])[TARGET].sum().rename("sold_to_date")
    inv = latest.set_index(["event_id", "tier"])[["capacity", "price"]].join(sold)
    inv["remaining"] = (inv["capacity"] - inv["sold_to_date"]).clip(lower=0).astype(int)
    return inv.rename(columns={"price": "current_price"})[["remaining", "current_price"]]


def fetch_inventory(client: InventoryClient, keys, fallback: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_id, tier in keys:
        try:
            item = client.get_inventory(event_id, tier)
            rows.append(
                {
                    "event_id": event_id,
                    "tier": tier,
                    "remaining": int(item["remaining"]),
                    "current_price": float(item["current_price"]),
                    "inventory_source": "api",
                }
            )
        except InventoryAPIError as exc:
            log.warning("inventory lookup failed for %s/%s, using sales history: %s", event_id, tier, exc)
            fb = fallback.loc[(event_id, tier)]
            rows.append(
                {
                    "event_id": event_id,
                    "tier": tier,
                    "remaining": int(fb["remaining"]),
                    "current_price": float(fb["current_price"]),
                    "inventory_source": "sales_history",
                }
            )
    return pd.DataFrame(rows)


def build_recommendations(
    scored: pd.DataFrame,
    history: pd.DataFrame,
    client: InventoryClient,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    keys = list(scored[["event_id", "tier"]].itertuples(index=False, name=None))
    inventory = fetch_inventory(client, keys, fallback_inventory(history))
    recs = scored.merge(inventory, on=["event_id", "tier"], how="left")

    days_left = (recs["event_date"] - as_of).dt.days
    recs["projected_sell_through"] = [
        projected_sell_through(d, n, r) for d, n, r in zip(recs["forecast_daily_demand"], days_left, recs["remaining"])
    ]
    recs["recommended_price"] = [
        recommend_price(p, st) for p, st in zip(recs["current_price"], recs["projected_sell_through"])
    ]
    recs["price_change_pct"] = (recs["recommended_price"] / recs["current_price"] - 1).round(4)
    recs["as_of"] = as_of.date()
    recs["event_date"] = recs["event_date"].dt.date
    recs["forecast_daily_demand"] = recs["forecast_daily_demand"].round(2)
    recs["projected_sell_through"] = recs["projected_sell_through"].round(3)
    return recs[OUTPUT_COLUMNS].sort_values(["event_id", "tier"]).reset_index(drop=True)
