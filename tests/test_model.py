import pytest

from pipeline.features import build_features
from pipeline.model import predict_demand, train_model, wape


def test_wape_perfect_forecast_is_zero():
    assert wape([5, 10, 0], [5, 10, 0]) == 0.0


def test_wape_hand_computed():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 20]
    # absolute errors 2 + 2 + 10 = 14
    assert wape(y_true, y_pred) == pytest.approx(0.28)


def test_train_model_reports_metrics(training_frame):
    model, metrics = train_model(build_features(training_frame))

    assert metrics["n_train"] > metrics["n_val"] > 0
    assert 0 <= metrics["val_wape"] < 1


def test_predictions_are_non_negative(training_frame):
    feats = build_features(training_frame)
    model, _ = train_model(feats)
    scored = predict_demand(model, feats.dropna())
    assert (scored["forecast_daily_demand"] >= 0).all()
