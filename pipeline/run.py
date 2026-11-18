"""Run the daily pipeline end to end.

    python -m pipeline.run                    # as of the latest day in the data
    python -m pipeline.run --as-of 2026-09-14
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.config import DATA_DIR, OUTPUT_DIR
from pipeline.features import build_features, make_scoring_frame
from pipeline.ingest import load_all
from pipeline.inventory_client import InventoryClient, MockInventoryAPI
from pipeline.model import predict_demand, train_model
from pipeline.pricing import build_recommendations

log = logging.getLogger("pipeline")


def write_recommendations(recs: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"recommendations_{date.today():%Y-%m-%d}.csv"
    recs.to_csv(path, index=False)
    return path


def run(data_dir: Path, output_dir: Path, as_of: pd.Timestamp | None = None) -> Path:
    history = load_all(data_dir, as_of)
    if as_of is None:
        as_of = history["date"].max()

    features = build_features(history)
    model, metrics = train_model(features)

    scored = predict_demand(model, make_scoring_frame(history, as_of))
    api = MockInventoryAPI(data_dir / "inventory_snapshot.json")
    recs = build_recommendations(scored, history, InventoryClient(api), as_of)

    path = write_recommendations(recs, output_dir)
    log.info(
        "%d recommendations for %d events (%d inventory API calls); median change %+.1f%%",
        len(recs),
        recs["event_id"].nunique(),
        api.calls,
        100 * recs["price_change_pct"].median(),
    )
    log.info("wrote %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast demand and recommend prices.")
    parser.add_argument("--as-of", type=pd.Timestamp, default=None, help="last day of data to use (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    run(args.data_dir, args.output_dir, args.as_of)


if __name__ == "__main__":
    main()
