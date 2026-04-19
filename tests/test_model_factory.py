import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.training.model_factory import build_model


def test_build_xgboost_model():
    cfg = {
        "type": "xgboost",
        "params": {"n_estimators": 10},
    }

    model = build_model(cfg)

    assert model is not None


def test_build_random_forest_model():
    cfg = {
        "type": "random_forest",
        "params": {"n_estimators": 10},
    }

    model = build_model(cfg)

    assert isinstance(model, RandomForestRegressor)


def test_build_linear_regression_model():
    cfg = {
        "type": "linear_regression",
        "params": {},
    }

    model = build_model(cfg)

    assert isinstance(model, LinearRegression)


def test_invalid_model_type():
    cfg = {
        "type": "invalid_model",
        "params": {},
    }

    with pytest.raises(ValueError):
        build_model(cfg)