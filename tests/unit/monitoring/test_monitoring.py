import pandas as pd

from src.monitoring.performance import compute_regression_metrics


def test_compute_regression_metrics():
    df = pd.DataFrame(
        {
            "Sales": [10, 20, 30],
            "prediction": [8, 22, 29],
        }
    )

    metrics = compute_regression_metrics(df)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "bias" in metrics
    assert metrics["n_samples"] == 3