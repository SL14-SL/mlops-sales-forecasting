import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.configs.loader import get_path, load_config
from src.constants import PROJECT_ROOT
from src.training.model_factory import build_model, fit_model
from src.training.target_transform import (
    inverse_transform_target,
    transform_target,
)
from src.training.train import normalize_feature_dtypes
from src.training.utils import build_drop_columns
from src.utils.logger import get_logger


logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def calculate_regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate regression metrics on the original target scale."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    errors = actual - predicted
    nonzero_mask = actual != 0

    rmse = float(
        np.sqrt(mean_squared_error(actual, predicted))
    )
    mae = float(mean_absolute_error(actual, predicted))

    actual_sum = float(np.abs(actual).sum())
    wmape = (
        float(np.abs(errors).sum() / actual_sum)
        if actual_sum > 0
        else float("nan")
    )

    rmspe = (
        float(
            np.sqrt(
                np.mean(
                    (
                        errors[nonzero_mask]
                        / actual[nonzero_mask]
                    )
                    ** 2
                )
            )
        )
        if nonzero_mask.any()
        else float("nan")
    )

    bias = float(errors.mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "wmape": wmape,
        "rmspe": rmspe,
        "bias": bias,
    }


def load_feature_data() -> pd.DataFrame:
    """Load the complete feature table used by the training pipeline."""
    features_path = f"{get_path('features')}/features.parquet"

    logger.info("Loading features from: %s", features_path)

    df = pd.read_parquet(features_path)

    data_cfg = TRAIN_CFG["data"]
    target_column = data_cfg["target_column"]
    time_column = data_cfg["time_column"]

    required_columns = [target_column, time_column]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Feature table is missing columns: {missing_columns}"
        )

    df = df.copy()
    df[time_column] = pd.to_datetime(
        df[time_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            time_column,
            target_column,
        ]
    )

    df = df.sort_values(time_column).reset_index(drop=True)

    logger.info(
        "Feature data loaded | rows=%s | start=%s | end=%s",
        len(df),
        df[time_column].min().date(),
        df[time_column].max().date(),
    )

    return df


def build_walk_forward_folds(
    df: pd.DataFrame,
    *,
    number_of_folds: int,
    validation_days: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build expanding-window folds using exact validation date blocks.

    Every validation block occurs strictly after its training block.
    """
    time_column = TRAIN_CFG["data"]["time_column"]

    unique_dates = pd.Index(
        sorted(df[time_column].dt.normalize().unique())
    )

    required_dates = number_of_folds * validation_days

    if len(unique_dates) <= required_dates:
        raise ValueError(
            "Not enough unique dates for the requested walk-forward folds."
        )

    validation_dates = unique_dates[-required_dates:]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold_index in range(number_of_folds):
        start_index = fold_index * validation_days
        end_index = start_index + validation_days

        fold_validation_dates = validation_dates[
            start_index:end_index
        ]

        validation_start = fold_validation_dates.min()
        validation_end = fold_validation_dates.max()

        train_df = df[
            df[time_column].dt.normalize() < validation_start
        ].copy()

        validation_df = df[
            df[time_column].dt.normalize().isin(
                fold_validation_dates
            )
        ].copy()

        if train_df.empty:
            raise ValueError(
                f"Training data is empty for fold {fold_index + 1}."
            )

        if validation_df.empty:
            raise ValueError(
                f"Validation data is empty for fold {fold_index + 1}."
            )

        logger.info(
            "Fold %s prepared | train_end=%s | "
            "validation_start=%s | validation_end=%s",
            fold_index + 1,
            train_df[time_column].max().date(),
            pd.Timestamp(validation_start).date(),
            pd.Timestamp(validation_end).date(),
        )

        folds.append((train_df, validation_df))

    return folds


def evaluate_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    fold_number: int,
) -> dict[str, float | int | str]:
    """Train and evaluate one walk-forward fold."""
    data_cfg = TRAIN_CFG["data"]
    model_cfg = TRAIN_CFG["model"]
    training_cfg = TRAIN_CFG.get("training", {})

    target_column = data_cfg["target_column"]
    time_column = data_cfg["time_column"]
    target_transformation = training_cfg.get(
        "target_transformation",
        "none",
    )

    model_type = model_cfg["type"]
    seed = ENV_CFG.get("random_seed")
    drop_columns = build_drop_columns(TRAIN_CFG)

    X_train = train_df.drop(
        columns=drop_columns,
        errors="ignore",
    )
    X_validation = validation_df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    X_train = normalize_feature_dtypes(X_train)
    X_validation = normalize_feature_dtypes(X_validation)

    y_train = transform_target(
        train_df[target_column],
        target_transformation,
    )
    y_validation_transformed = transform_target(
        validation_df[target_column],
        target_transformation,
    )

    model = build_model(
        model_cfg,
        seed=seed,
    )

    logger.info(
        "Training fold %s | train_rows=%s | validation_rows=%s",
        fold_number,
        len(train_df),
        len(validation_df),
    )

    fit_model(
        model=model,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_validation,
        y_val=y_validation_transformed,
    )

    predictions_transformed = model.predict(X_validation)
    predictions = inverse_transform_target(
        predictions_transformed,
        target_transformation,
    )

    actual = validation_df[target_column].to_numpy()

    metrics = calculate_regression_metrics(
        actual,
        predictions,
    )

    result: dict[str, float | int | str] = {
        "fold": fold_number,
        "train_start": str(train_df[time_column].min().date()),
        "train_end": str(train_df[time_column].max().date()),
        "validation_start": str(
            validation_df[time_column].min().date()
        ),
        "validation_end": str(
            validation_df[time_column].max().date()
        ),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        **metrics,
    }

    logger.info(
        "Fold %s completed | RMSE=%.2f | MAE=%.2f | "
        "WMAPE=%.4f | RMSPE=%.4f | Bias=%.2f",
        fold_number,
        metrics["rmse"],
        metrics["mae"],
        metrics["wmape"],
        metrics["rmspe"],
        metrics["bias"],
    )

    return result


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create an aggregate metric summary across all folds."""
    metric_columns = [
        "rmse",
        "mae",
        "wmape",
        "rmspe",
        "bias",
    ]

    summary_rows = []

    for metric in metric_columns:
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(results_df[metric].mean()),
                "std": float(
                    results_df[metric].std(ddof=0)
                ),
                "min": float(results_df[metric].min()),
                "max": float(results_df[metric].max()),
            }
        )

    return pd.DataFrame(summary_rows)


def run_backtest(
    *,
    number_of_folds: int,
    validation_days: int,
    output_directory: Path,
) -> None:
    """Run the complete walk-forward model backtest."""
    df = load_feature_data()

    folds = build_walk_forward_folds(
        df,
        number_of_folds=number_of_folds,
        validation_days=validation_days,
    )

    results = []

    for fold_number, (train_df, validation_df) in enumerate(
        folds,
        start=1,
    ):
        result = evaluate_fold(
            train_df,
            validation_df,
            fold_number=fold_number,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_output_path = (
        output_directory / "model_backtest_folds.csv"
    )
    summary_output_path = (
        output_directory / "model_backtest_summary.csv"
    )

    results_df.to_csv(
        fold_output_path,
        index=False,
    )
    summary_df.to_csv(
        summary_output_path,
        index=False,
    )

    logger.info(
        "Backtest completed | folds=%s | mean_rmse=%.2f | "
        "mean_mae=%.2f | mean_wmape=%.4f",
        number_of_folds,
        results_df["rmse"].mean(),
        results_df["mae"].mean(),
        results_df["wmape"].mean(),
    )

    logger.info(
        "Fold results saved to: %s",
        fold_output_path,
    )
    logger.info(
        "Summary saved to: %s",
        summary_output_path,
    )

    print()
    print("Walk-forward fold results")
    print(results_df.to_string(index=False))

    print()
    print("Aggregate summary")
    print(summary_df.to_string(index=False))


def main() -> None:
    """Parse command-line arguments and run the backtest."""
    parser = argparse.ArgumentParser(
        description="Run an expanding-window forecasting backtest."
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of walk-forward folds.",
    )

    parser.add_argument(
        "--validation-days",
        type=int,
        default=14,
        help="Number of unique dates in each validation fold.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory used for backtest result files.",
    )

    args = parser.parse_args()

    if args.folds < 1:
        raise ValueError("--folds must be at least 1.")

    if args.validation_days < 1:
        raise ValueError(
            "--validation-days must be at least 1."
        )

    run_backtest(
        number_of_folds=args.folds,
        validation_days=args.validation_days,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()