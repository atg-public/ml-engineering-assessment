import numpy as np
import pandas as pd

from pipeline.features import FEATURES, build_features, make_scoring_frame, training_rows


def test_calendar_features(daily_frame):
    feats = build_features(daily_frame)
    first = feats[(feats["event_id"] == "201") & (feats["date"] == "2026-08-01")].iloc[0]

    assert first["dow"] == 5  # Saturday
    assert first["is_weekend"] == 1
    assert first["days_to_event"] == 31
    assert first["tier_code"] == 0


def test_lags_do_not_cross_series(daily_frame):
    feats = build_features(daily_frame)
    for _, series in feats.groupby("event_id"):
        assert np.isnan(series["sales_lag_1"].iloc[0])
        assert series["sales_lag_1"].iloc[1:].tolist() == series["tickets_sold"].iloc[:-1].tolist()
        assert series["sales_lag_7"].notna().sum() == len(series) - 7


def test_sessions_roll_uses_previous_week(daily_frame):
    feats = build_features(daily_frame)
    series = feats[feats["event_id"] == "201"].reset_index(drop=True)
    # day index 7: previous 7 days of sessions are 100..160
    assert series.loc[7, "sessions_roll_7"] == np.mean([100, 110, 120, 130, 140, 150, 160])


def test_training_rows_have_complete_features(daily_frame):
    rows = training_rows(build_features(daily_frame))
    assert len(rows) == 2 * 3  # 10 days minus the 7 needed for sales_lag_7
    assert rows[FEATURES].notna().all().all()


def test_scoring_frame_is_one_row_per_series_for_next_day(daily_frame):
    as_of = pd.Timestamp("2026-08-10")
    scoring = make_scoring_frame(daily_frame, as_of)

    assert len(scoring) == 2
    assert (scoring["date"] == as_of + pd.Timedelta(days=1)).all()
    assert scoring["sales_lag_1"].tolist() == [10.0, 10.0]
