import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from unittest.mock import MagicMock

from src.training.model_factory import fit_model, build_model


def test_fit_xgboost_candidate_uses_validation_data():
    model = MagicMock()

    fit_model(
        model=model,
        model_type="xgboost",
        X_train="X_train",
        y_train="y_train",
        X_val="X_val",
        y_val="y_val",
        sample_weight="weights",
    )

    model.fit.assert_called_once_with(
        X="X_train",
        y="y_train",
        sample_weight="weights",
        eval_set=[("X_val", "y_val")],
        verbose=False,
    )


def test_fit_xgboost_final_refit_uses_all_data_without_validation():
    model = MagicMock()

    fit_model(
        model=model,
        model_type="xgboost",
        X_train="X_all",
        y_train="y_all",
        sample_weight="weights",
    )

    model.fit.assert_called_once_with(
        X="X_all",
        y="y_all",
        sample_weight="weights",
        verbose=False,
    )

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