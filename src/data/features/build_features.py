import pandas as pd

from src.configs.loader import load_config
from src.data.features.common import (
    cast_object_columns_to_category,
    drop_columns_if_present,
)
from src.data.features.core import (
    add_temporal_features,
    add_training_lag_features,
    initialize_inference_lag_placeholders,
    sort_frame,
)

from src.data.features.forecasting_policy import (
    FORECASTING_TECHNICAL_DROP_COLUMNS,
    add_competition_duration_features,
    add_promo_duration_features,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)
TRAIN_CFG = load_config("training.yaml")

def _get_data_config(config: dict) -> dict:
    data_cfg = config.get("data", {})
    if not data_cfg:
        raise ValueError("Missing 'data' section in config.")
    return data_cfg


def _get_feature_config(config: dict) -> dict:
    return config.get("features", {})


def _resolve_core_columns(config: dict) -> dict:
    data_cfg = _get_data_config(config)

    id_columns = data_cfg.get("id_columns", [])
    if not id_columns:
        raise ValueError("Config must define at least one id column in data.id_columns.")

    return {
        "entity_column": id_columns[0],
        "target_column": data_cfg["target_column"],
        "date_column": data_cfg["time_column"],
    }


def _infer_mode(df: pd.DataFrame, mode: str, target_column: str) -> tuple[bool, bool]:
    if mode not in {"auto", "train", "inference"}:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    is_training_mode = mode == "train" or (mode == "auto" and target_column in df.columns)
    is_inference_mode = mode == "inference" or (mode == "auto" and target_column not in df.columns)

    return is_training_mode, is_inference_mode

def _get_lag_config(config: dict) -> dict:
    return config.get("features", {}).get(
        "lag_features",
        {
            "lags": [1, 7],
            "rolling_windows": [7],
        },
    )

def _apply_step(
    df: pd.DataFrame,
    *,
    step_name: str,
    config: dict,
    is_training_mode: bool,
    is_inference_mode: bool,
    entity_column: str,
    target_column: str,
    date_column: str,
) -> pd.DataFrame:
    """
    Apply one configured feature-engineering step.

    Args:
        df: Input dataframe for the current pipeline step.
        step_name: Name of the configured transformation.
        config: Effective training and feature configuration.
        is_training_mode: Whether target-derived training features may be created.
        is_inference_mode: Whether inference placeholders must be initialized.
        entity_column: Column identifying the forecasting entity.
        target_column: Forecast target column.
        date_column: Temporal ordering column.

    Returns:
        A transformed copy of the dataframe.

    Raises:
        ValueError: If the configured feature step is unknown.
    """
    if step_name == "sort":
        return sort_frame(
            df,
            entity_column=entity_column,
            date_column=date_column,
        )

    if step_name == "temporal":
        return add_temporal_features(
            df,
            date_column=date_column,
        )

    if step_name == "lags":
        lag_cfg = _get_lag_config(config)
        lags = lag_cfg.get("lags", [1, 7])
        rolling_windows = lag_cfg.get("rolling_windows", [7])

        if is_training_mode:
            return add_training_lag_features(
                df,
                entity_column=entity_column,
                target_column=target_column,
                lags=lags,
                rolling_windows=rolling_windows,
            )

        if is_inference_mode:
            return initialize_inference_lag_placeholders(
                df,
                target_column=target_column,
                lags=lags,
                rolling_windows=rolling_windows,
            )

        return df

    if step_name == "competition":
        return add_competition_duration_features(df)

    if step_name == "promo":
        return add_promo_duration_features(
            df,
            entity_column=entity_column,
            date_column=date_column,
            promo_column="Promo",
        )

    if step_name == "cast_categoricals":
        return cast_object_columns_to_category(df)

    if step_name == "drop_technical":
        technical_drop_columns = config.get("features", {}).get(
            "technical_drop_columns",
            FORECASTING_TECHNICAL_DROP_COLUMNS,
        )
        return drop_columns_if_present(df, technical_drop_columns)

    if step_name == "drop_configured":
        configured_drop_columns = config.get("features", {}).get("drop_columns", [])
        return drop_columns_if_present(df, configured_drop_columns)

    raise ValueError(f"Unknown feature step configured: {step_name}")


def build_features(
    df: pd.DataFrame,
    config: dict | None = None,
    *,
    mode: str = "auto",
) -> pd.DataFrame:
    """
    Build configured forecasting features for training or inference.

    The effective mode controls whether target-derived lag features are calculated
    or only inference placeholders are initialized. Feature steps are executed in
    their configured order.

    Args:
        df: Source dataframe containing the configured entity and date columns.
        config: Optional feature configuration. Defaults to the training config.
        mode: One of ``"auto"``, ``"train"`` or ``"inference"``.

    Returns:
        A transformed dataframe containing the configured model features.

    Raises:
        ValueError: If the mode, core-column configuration or a feature step is
            invalid.
    """
    config = config or TRAIN_CFG

    if df.empty:
        logger.info("Received empty dataframe in build_features(). Returning unchanged dataframe.")
        return df.copy()

    df = df.copy()

    columns = _resolve_core_columns(config)
    entity_column = columns["entity_column"]
    target_column = columns["target_column"]
    date_column = columns["date_column"]

    is_training_mode, is_inference_mode = _infer_mode(df, mode, target_column)

    feature_cfg = _get_feature_config(config)
    enabled_steps = feature_cfg.get(
        "enabled_steps",
        [
            "sort",
            "temporal",
            "lags",
            "cast_categoricals",
            "drop_technical",
            "drop_configured",
        ],
    )

    logger.info(
        "Building features | "
        f"rows={len(df)} | "
        f"mode={mode} | "
        f"training_mode={is_training_mode} | "
        f"steps={enabled_steps}"
    )

    for step_name in enabled_steps:
        df = _apply_step(
            df,
            step_name=step_name,
            config=config,
            is_training_mode=is_training_mode,
            is_inference_mode=is_inference_mode,
            entity_column=entity_column,
            target_column=target_column,
            date_column=date_column,
        )

    return df


def preprocess_data(df: pd.DataFrame, *, mode: str = "auto") -> pd.DataFrame:
    """
    Build features using the default training configuration.
    """
    return build_features(df, config=TRAIN_CFG, mode=mode)
