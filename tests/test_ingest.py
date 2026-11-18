import pandas as pd

from pipeline.ingest import build_daily_frame, load_sales


def test_load_sales_collapses_to_one_row_per_day(tmp_path):
    pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-01", "2026-08-02"],
            "event_id": [101, 101, 101],
            "tier": ["A", "A", "A"],
            "price": [90.0, 90.0, 95.0],
            "tickets_sold": [4, 2, 5],
        }
    ).to_csv(tmp_path / "sales.csv", index=False)

    daily = load_sales(tmp_path)

    assert len(daily) == 2
    assert daily.loc[daily["date"] == "2026-08-01", "tickets_sold"].item() == 6
    assert daily.loc[daily["date"] == "2026-08-02", "price"].item() == 95.0


def test_build_daily_frame_attaches_events_and_traffic():
    sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-02"]),
            "event_id": [101, 101],
            "tier": ["B", "B"],
            "tickets_sold": [3, 7],
            "price": [55.0, 55.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [101],
            "event_name": ["The Paper Comet"],
            "genre": ["play"],
            "event_date": pd.to_datetime(["2026-10-01"]),
            "tier": ["B"],
            "capacity": [400],
            "base_price": [55.0],
        }
    )
    traffic = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-02"]),
            "event_id": ["101", "101"],
            "sessions": [120, 180],
            "page_views": [300, 410],
        }
    )

    df = build_daily_frame(sales, traffic, events)

    assert len(df) == 2
    assert df["capacity"].tolist() == [400, 400]
    assert df["sessions"].tolist() == [120, 180]
