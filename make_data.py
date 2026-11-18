"""Generate the synthetic dataset used by the pipeline.

Everything here is invented: event names, prices, capacities and web traffic
are drawn from fixed distributions with a fixed seed, so every run writes
identical files.

    python make_data.py              # writes ./data
    python make_data.py --out other  # writes somewhere else
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_EVENTS = 32
FIRST_DAY = pd.Timestamp("2026-07-01")
N_DAYS = 90
TIERS = ("A", "B", "C")
TIER_PRICE_RANGE = {"A": (85, 130), "B": (50, 80), "C": (22, 40)}
TIER_SHARE = {"A": 0.20, "B": 0.35, "C": 0.45}
CONVERSION = 0.04
ELASTICITY = -1.3
DOW_EFFECT = np.array([0.85, 0.80, 0.90, 1.00, 1.25, 1.35, 1.05])

ADJECTIVES = [
    "Glass", "Velvet", "Hollow", "Silver", "Paper", "Crimson", "Quiet", "Electric",
    "Wandering", "Northern", "Borrowed", "Golden", "Lantern", "Midnight", "Copper", "Restless",
]
NOUNS = [
    "Orchard", "Lighthouse", "Cartographer", "Parade", "Harbour", "Machine", "Garden", "Tide",
    "Circus", "Archive", "Comet", "Kingdom", "Ballroom", "Signal", "Meadow", "Engine",
]
GENRES = ["musical", "play", "comedy", "concert", "dance"]


def round_half(x: float) -> float:
    return round(x * 2) / 2


def price_path(rng: np.random.Generator, base: float, n: int) -> np.ndarray:
    """Step-wise price history: occasional manual changes of 4-12%."""
    prices = np.empty(n)
    current = base
    for t in range(n):
        if t > 0 and rng.random() < 0.05:
            step = rng.uniform(0.04, 0.12) * rng.choice([-1, 1])
            current = round_half(current * (1 + step))
        prices[t] = current
    return prices


def event_names(rng: np.random.Generator, n: int) -> list[str]:
    pairs = [(a, b) for a in ADJECTIVES for b in NOUNS]
    picks = rng.choice(len(pairs), size=n, replace=False)
    return [f"The {pairs[i][0]} {pairs[i][1]}" for i in picks]


def simulate(rng: np.random.Generator):
    dates = pd.date_range(FIRST_DAY, periods=N_DAYS, freq="D")
    last_day = dates[-1]
    names = event_names(rng, N_EVENTS)
    event_ids = [101 + i for i in range(N_EVENTS)]

    breakout = int(rng.integers(N_EVENTS))
    others = [e for e in event_ids if e != event_ids[breakout]]
    # Events that moved to the new website are tracked by the new analytics
    # property, which exports its own content IDs.
    new_site = set(rng.choice(others, size=5, replace=False).tolist())
    # One show has moved to a partner's box office, so the inventory service
    # no longer lists it.
    partner_run = int(rng.choice([e for e in others if e not in new_site]))

    events, sales, traffic, inventory = [], [], [], []
    for i, event_id in enumerate(event_ids):
        popularity = rng.lognormal(0.0, 0.45)
        genre = GENRES[int(rng.integers(len(GENRES)))]
        event_date = last_day + pd.Timedelta(days=int(rng.integers(21, 110)))
        days_out = (event_date - dates).days.to_numpy()
        build = 1 + 1.2 * np.exp(-days_out / 35)

        sessions = 300 * popularity * DOW_EFFECT[dates.dayofweek] * build
        sessions = sessions * rng.lognormal(0.0, 0.15, N_DAYS)
        for day in rng.choice(N_DAYS, size=3, replace=False):
            sessions[day : day + 2] *= 1.8  # marketing pushes
        if i == breakout:
            sessions[45:] *= 2.6  # strong reviews; demand steps up and stays up
        sessions = np.round(sessions).astype(int)
        page_views = np.round(sessions * rng.uniform(2.2, 3.4, N_DAYS)).astype(int)

        traffic_key = f"{event_id:06d}" if event_id in new_site else event_id
        traffic.append(
            pd.DataFrame(
                {"date": dates, "event_id": traffic_key, "sessions": sessions, "page_views": page_views}
            )
        )

        for tier in TIERS:
            base = round_half(rng.uniform(*TIER_PRICE_RANGE[tier]))
            prices = price_path(rng, base, N_DAYS)
            demand = sessions * CONVERSION * TIER_SHARE[tier] * (prices / base) ** ELASTICITY
            sold = rng.poisson(demand)

            # Size each tier so it's on course to sell 55-135% of what's left;
            # the breakout show's premium tier is nearly gone.
            days_left = (event_date - last_day).days
            pace = 9.7 if (i == breakout and tier == "A") else rng.uniform(0.55, 1.35)
            remaining = max(int(sold[-14:].mean() * days_left / pace), 5)
            capacity = int(sold.sum()) + remaining
            holds = int(rng.integers(0, 6))

            events.append(
                {
                    "event_id": event_id,
                    "event_name": names[i],
                    "genre": genre,
                    "event_date": event_date.date().isoformat(),
                    "tier": tier,
                    "capacity": capacity,
                    "base_price": base,
                }
            )
            sales.append(
                pd.DataFrame(
                    {"date": dates, "event_id": event_id, "tier": tier, "price": prices, "tickets_sold": sold}
                )
            )
            if event_id != partner_run:
                inventory.append(
                    {
                        "event_id": event_id,
                        "tier": tier,
                        "remaining": max(remaining - holds, 0),
                        "current_price": float(prices[-1]),
                    }
                )

    return (
        pd.DataFrame(events),
        pd.concat(sales, ignore_index=True),
        pd.concat(traffic, ignore_index=True),
        {"as_of": last_day.date().isoformat(), "items": inventory},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    events, sales, traffic, inventory = simulate(rng)

    events.to_csv(args.out / "events.csv", index=False)

    # The box office delivers history as a backfill extract plus an incremental one.
    cutoff = FIRST_DAY + pd.Timedelta(days=45)
    backfill = sales[sales["date"] <= cutoff]
    incremental = sales[sales["date"] >= cutoff]
    pd.concat([backfill, incremental]).to_csv(args.out / "sales.csv", index=False, date_format="%Y-%m-%d")

    traffic.sort_values(["date"], kind="stable").to_csv(
        args.out / "web_traffic.csv", index=False, date_format="%Y-%m-%d"
    )
    (args.out / "inventory_snapshot.json").write_text(json.dumps(inventory, indent=2))

    print(f"wrote {len(events)} event/tier rows, {len(sales)} sales days, {len(traffic)} traffic days to {args.out}")


if __name__ == "__main__":
    main()
