import argparse
import gc
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from run_model_backtest import (
    build_walk_forward_folds,
    calculate_regression_metrics,
    load_feature_data,
)
from src.configs.loader import load_config
from src.constants import PROJECT_ROOT
from src.training.model_factory import (
    build_model,
    fit_model,
)
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


def build_trial_model_config(
    trial: optuna.Trial,
) -> dict:
    """Build an XGBoost configuration for one Optuna trial."""
    model_config = deepcopy(
        TRAIN_CFG["model"]
    )
    base_params = deepcopy(
        model_config.get("params", {})
    )

    trial_params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_categorical(
            "n_estimators",
            [
                300,
                500,
                700,
                900,
                1200,
            ],
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            4,
            9,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.02,
            0.10,
            log=True,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            15,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.70,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.70,
            1.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-4,
            1.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1.0,
            20.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            0.5,
        ),
        "early_stopping_rounds": 50,
    }

    base_params.update(trial_params)

    model_config["params"] = base_params
    return model_config


def evaluate_model_config_on_fold(
    model_config: dict,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, float]:
    """Train and evaluate one model configuration on one fold."""
    data_config = TRAIN_CFG["data"]
    training_config = TRAIN_CFG.get(
        "training",
        {},
    )

    target_column = data_config["target_column"]
    target_transformation = training_config.get(
        "target_transformation",
        "none",
    )
    model_type = model_config["type"]
    drop_columns = build_drop_columns(
        TRAIN_CFG
    )
    seed = ENV_CFG.get("random_seed")

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(
        drop=True
    )

    X_train = train_df.drop(
        columns=drop_columns,
        errors="ignore",
    )
    X_validation = validation_df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    X_train = normalize_feature_dtypes(
        X_train
    )
    X_validation = normalize_feature_dtypes(
        X_validation
    )

    y_train = transform_target(
        train_df[target_column],
        target_transformation,
    )
    y_validation = transform_target(
        validation_df[target_column],
        target_transformation,
    )

    model = build_model(
        model_config,
        seed=seed,
    )

    fit_model(
        model=model,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_validation,
        y_val=y_validation,
    )

    transformed_predictions = model.predict(
        X_validation
    )
    predictions = inverse_transform_target(
        transformed_predictions,
        target_transformation,
    )

    metrics = calculate_regression_metrics(
        validation_df[target_column],
        predictions,
    )

    if hasattr(model, "best_iteration"):
        metrics["best_iteration"] = float(
            model.best_iteration
        )

    del model
    del X_train
    del X_validation
    gc.collect()

    return metrics


def create_objective(
    selected_folds: list[
        tuple[int, pd.DataFrame, pd.DataFrame]
    ],
):
    """Create the Optuna objective using selected walk-forward folds."""

    def objective(
        trial: optuna.Trial,
    ) -> float:
        model_config = build_trial_model_config(
            trial
        )

        fold_rmses = []
        fold_maes = []
        fold_wmapes = []
        best_iterations = []

        for (
            fold_number,
            train_df,
            validation_df,
        ) in selected_folds:
            logger.info(
                "Trial %s | training fold %s",
                trial.number,
                fold_number,
            )

            metrics = evaluate_model_config_on_fold(
                model_config,
                train_df,
                validation_df,
            )

            fold_rmses.append(metrics["rmse"])
            fold_maes.append(metrics["mae"])
            fold_wmapes.append(metrics["wmape"])

            if "best_iteration" in metrics:
                best_iterations.append(
                    metrics["best_iteration"]
                )

            trial.set_user_attr(
                f"fold_{fold_number}_rmse",
                metrics["rmse"],
            )
            trial.set_user_attr(
                f"fold_{fold_number}_mae",
                metrics["mae"],
            )
            trial.set_user_attr(
                f"fold_{fold_number}_wmape",
                metrics["wmape"],
            )

        mean_rmse = float(
            np.mean(fold_rmses)
        )
        mean_mae = float(
            np.mean(fold_maes)
        )
        mean_wmape = float(
            np.mean(fold_wmapes)
        )

        trial.set_user_attr(
            "mean_mae",
            mean_mae,
        )
        trial.set_user_attr(
            "mean_wmape",
            mean_wmape,
        )

        if best_iterations:
            trial.set_user_attr(
                "mean_best_iteration",
                float(
                    np.mean(best_iterations)
                ),
            )

        logger.info(
            "Trial %s completed | "
            "mean_rmse=%.2f | mean_mae=%.2f | "
            "mean_wmape=%.4f",
            trial.number,
            mean_rmse,
            mean_mae,
            mean_wmape,
        )

        return mean_rmse

    return objective


def save_study_results(
    study: optuna.Study,
    output_directory: Path,
) -> None:
    """Persist Optuna trials and best parameters."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_path = (
        output_directory
        / "hyperparameter_trials.csv"
    )
    best_params_path = (
        output_directory
        / "best_hyperparameters.json"
    )

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "user_attrs",
            "state",
        )
    )
    trials_df.to_csv(
        trials_path,
        index=False,
    )

    best_result = {
        "best_trial": study.best_trial.number,
        "mean_rmse": study.best_value,
        "parameters": study.best_params,
        "user_attributes": (
            study.best_trial.user_attrs
        ),
    }

    with best_params_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_result,
            file,
            indent=2,
        )

    logger.info(
        "Tuning trials saved to: %s",
        trials_path,
    )
    logger.info(
        "Best parameters saved to: %s",
        best_params_path,
    )


def run_tuning(
    *,
    number_of_trials: int,
    selected_fold_numbers: list[int],
    validation_days: int,
    output_directory: Path,
) -> None:
    """Run the complete hyperparameter search."""
    feature_df = load_feature_data()

    all_folds = build_walk_forward_folds(
        feature_df,
        number_of_folds=4,
        validation_days=validation_days,
    )

    invalid_folds = [
        fold_number
        for fold_number in selected_fold_numbers
        if fold_number < 1
        or fold_number > len(all_folds)
    ]

    if invalid_folds:
        raise ValueError(
            f"Invalid fold numbers: {invalid_folds}"
        )

    selected_folds = [
        (
            fold_number,
            all_folds[fold_number - 1][0],
            all_folds[fold_number - 1][1],
        )
        for fold_number in selected_fold_numbers
    ]

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_path = (
        output_directory
        / "optuna_forecasting.db"
    )
    storage_uri = (
        f"sqlite:///{storage_path.resolve()}"
    )

    study = optuna.create_study(
        study_name="forecasting-xgboost-tuning",
        direction="minimize",
        storage=storage_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=ENV_CFG.get(
                "random_seed",
                42,
            ),
        ),
    )

    current_params = TRAIN_CFG[
        "model"
    ].get("params", {})

    study.enqueue_trial(
        {
            "n_estimators": current_params.get(
                "n_estimators",
                300,
            ),
            "max_depth": current_params.get(
                "max_depth",
                8,
            ),
            "learning_rate": current_params.get(
                "learning_rate",
                0.05,
            ),
            "min_child_weight": current_params.get(
                "min_child_weight",
                1,
            ),
            "subsample": current_params.get(
                "subsample",
                0.8,
            ),
            "colsample_bytree": current_params.get(
                "colsample_bytree",
                1.0,
            ),
            "reg_alpha": current_params.get(
                "reg_alpha",
                1e-4,
            ),
            "reg_lambda": current_params.get(
                "reg_lambda",
                1.0,
            ),
            "gamma": current_params.get(
                "gamma",
                0.0,
            ),
        }
    )

    logger.info(
        "Starting tuning | trials=%s | folds=%s",
        number_of_trials,
        selected_fold_numbers,
    )

    study.optimize(
        create_objective(
            selected_folds
        ),
        n_trials=number_of_trials,
        gc_after_trial=True,
    )

    save_study_results(
        study,
        output_directory,
    )

    print()
    print("Best trial")
    print(f"Trial: {study.best_trial.number}")
    print(
        f"Mean RMSE: {study.best_value:.2f}"
    )
    print(
        json.dumps(
            study.best_params,
            indent=2,
        )
    )


def main() -> None:
    """Parse command-line arguments and run tuning."""
    parser = argparse.ArgumentParser(
        description=(
            "Tune XGBoost on selected "
            "walk-forward folds."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[
            2,
            3,
        ],
        help="Fold numbers used during tuning.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=14,
        help="Validation days per fold.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "hyperparameter_tuning"
        ),
        help="Directory for tuning outputs.",
    )

    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError(
            "--trials must be at least 1."
        )

    run_tuning(
        number_of_trials=args.trials,
        selected_fold_numbers=args.folds,
        validation_days=args.validation_days,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()