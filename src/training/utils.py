from src.utils.logger import get_logger
from src.configs.loader import load_config


logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def build_drop_columns(config: dict) -> list[str]:
    """Build feature drop list from config without duplicates."""
    data_cfg = TRAIN_CFG.get("data", {})
    feature_cfg = TRAIN_CFG.get("features", {})

    target_column = data_cfg["target_column"]
    known_targets = data_cfg.get("known_targets", [])
    time_column = data_cfg.get("time_column")
    configured_drop_columns = feature_cfg.get("drop_columns", [])

    drop_columns = configured_drop_columns + known_targets + [target_column]

    if time_column:
        drop_columns.append(time_column)

    return list(dict.fromkeys(drop_columns))
