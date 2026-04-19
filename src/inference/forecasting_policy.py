import pandas as pd

from src.configs.loader import load_config
from src.data.features.build_features import preprocess_data
from src.data.features.core import get_lag_feature_names


CFG = load_config("training.yaml")
TARGET_COLUMN = CFG["data"]["target_column"]
ENTITY_COLUMN = CFG["data"]["id_columns"][0]
DATE_COLUMN = CFG["data"]["time_column"]


def normalize_store_key(validated_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize forecasting entity key used for metadata merge and state lookup."""
    validated_df = validated_df.copy()

    if ENTITY_COLUMN in validated_df.columns:
        validated_df[ENTITY_COLUMN] = validated_df[ENTITY_COLUMN].astype(int)

    return validated_df


def merge_request_with_metadata(
    validated_df: pd.DataFrame,
    store_metadata: pd.DataFrame,
    store_id: int,
) -> pd.DataFrame:
    """Merge request payload with static store metadata."""
    features_df = validated_df.merge(store_metadata, on=ENTITY_COLUMN, how="left")

    if "Promo2" not in features_df.columns or features_df["Promo2"].isnull().any():
        raise ValueError(
            f"Metadata missing for {ENTITY_COLUMN}={store_id}. Check store.parquet content."
        )

    return features_df


def run_forecasting_feature_engineering(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply project-specific forecasting feature engineering."""
    return preprocess_data(features_df, mode="inference")


def _build_state_feature_table(
    entity_ids: list[int],
    store_state: dict,
) -> pd.DataFrame:
    """
    Build one lag/rolling feature row per entity/store.

    This mirrors the training semantics from add_training_lag_features():
    - lag_k = shift(k), else 0 if insufficient history
    - rolling_mean_w = mean of the previous w values only if at least w values exist
    - otherwise 0

    Assumption:
    store_state[entity_id] contains past target values in chronological order
    (oldest -> newest), and does NOT include the current target to be predicted.
    """
    lag_cfg = CFG.get("features", {}).get("lag_features", {})
    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    feature_names = get_lag_feature_names(
        TARGET_COLUMN,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    rows: list[dict] = []

    for entity_id in entity_ids:
        raw_history = store_state.get(str(entity_id), [])
        history = [float(v) for v in raw_history] if raw_history else []

        row = {ENTITY_COLUMN: int(entity_id)}

        # Mirrors groupby.shift(lag).fillna(0)
        for lag in lags:
            key = f"lag_{lag}"
            row[feature_names[key]] = (
                float(history[-lag]) if len(history) >= lag else 0.0
            )

        # Mirrors x.shift(1).rolling(window=window).mean().fillna(0)
        # Since history already represents values BEFORE the prediction date,
        # we simply take the last `window` values if enough exist.
        for window in rolling_windows:
            key = f"rolling_mean_{window}"
            if len(history) >= window:
                values = history[-window:]
                row[feature_names[key]] = float(sum(values) / len(values))
            else:
                row[feature_names[key]] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def inject_forecasting_state_features(
    processed_df: pd.DataFrame,
    store_state: dict,
    store_id: int,
) -> pd.DataFrame:
    """Inject lag-based forecasting features from latest state snapshot."""
    processed_df = processed_df.copy()

    lag_cfg = CFG.get("features", {}).get("lag_features", {})
    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    feature_names = get_lag_feature_names(
        TARGET_COLUMN,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    history = store_state.get(str(store_id), [])

    if not history:
        history = [0.0] * 7

    if len(history) < 7:
        history = [0.0] * (7 - len(history)) + list(history)

    for lag in lags:
        key = f"lag_{lag}"
        processed_df[feature_names[key]] = (
            float(history[-lag]) if len(history) >= lag else 0.0
        )

    for window in rolling_windows:
        key = f"rolling_mean_{window}"
        values = history[-window:] if len(history) >= window else history
        if not values:
            processed_df[feature_names[key]] = 0.0
        else:
            processed_df[feature_names[key]] = float(sum(values) / len(values))

    return processed_df

def finalize_forecasting_feature_frame(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Finalize forecasting feature frame before model inference."""
    processed_df = processed_df.copy()

    if DATE_COLUMN in processed_df.columns:
        processed_df = processed_df.drop(columns=[DATE_COLUMN])

    return processed_df


def apply_forecasting_business_rules(prediction: float, is_open: int) -> float:
    """Apply forecasting-specific post-processing rules."""
    if is_open == 0:
        return 0.0
    return max(0.0, prediction)