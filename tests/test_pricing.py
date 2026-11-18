import pytest

from pipeline.config import TARGET_SELL_THROUGH
from pipeline.pricing import projected_sell_through, recommend_price


def test_projected_sell_through():
    assert projected_sell_through(daily_demand=5, days_to_event=20, remaining=200) == pytest.approx(0.5)


def test_projected_sell_through_handles_sold_out():
    assert projected_sell_through(daily_demand=5, days_to_event=20, remaining=0) == pytest.approx(100)


def test_on_target_keeps_price():
    assert recommend_price(60.0, TARGET_SELL_THROUGH) == 60.0


def test_selling_ahead_raises_price():
    assert recommend_price(60.0, 1.2) > 60.0


def test_selling_behind_lowers_price():
    assert recommend_price(60.0, 0.4) < 60.0
