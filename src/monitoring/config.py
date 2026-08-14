# src/monitoring/config.py
from src.configs.loader import load_config

def get_monitoring_config() -> dict:
    return load_config("monitoring.yaml")

def get_feature_drift_settings() -> dict:
    cfg = get_monitoring_config().get("feature_drift", {})
    return {
        "enabled": cfg.get("enabled", True),
        "numeric_features": cfg.get("numeric_features", []),
        "categorical_features": cfg.get("categorical_features", []),
        "min_samples": cfg.get("min_samples", 50),
        "p_value_threshold": cfg.get("p_value_threshold", 0.01),
        "stat_threshold": cfg.get("stat_threshold", 0.10),
    }

def get_data_quality_settings() -> dict:
    cfg = get_monitoring_config().get("data_quality", {})
    return {
        "enabled": cfg.get("enabled", True),
        "categorical_reference_features": cfg.get(
            "categorical_reference_features", []
        ),
        "persist_history": cfg.get("persist_history", False),
    }

def get_serving_settings() -> dict:
    cfg = get_monitoring_config().get("serving", {})
    return {
        "enabled": cfg.get("enabled", True),
        "metrics_endpoint_enabled": cfg.get("metrics_endpoint_enabled", True),
        "summary_endpoint_enabled": cfg.get("summary_endpoint_enabled", True),
        "summary_window_seconds": cfg.get("summary_window_seconds", 900),
        "track_paths": cfg.get("track_paths", ["/predict", "/health"]),
        "ignored_paths": cfg.get(
            "ignored_paths",
            ["/metrics", "/monitoring/summary", "/docs", "/openapi.json", "/redoc"],
        ),
    }

def get_retraining_settings() -> dict:
    cfg = get_monitoring_config().get(
        "retraining",
        {},
    )

    drift_cfg = cfg.get("drift", {})
    performance_cfg = cfg.get(
        "performance",
        {},
    )

    return {
        "minimum_new_training_rows": int(
            cfg.get(
                "minimum_new_training_rows",
                500,
            )
        ),
        "cooldown_hours": int(
            cfg.get(
                "cooldown_hours",
                168,
            )
        ),
        "scheduled_interval_hours": int(
            cfg.get(
                "scheduled_interval_hours",
                168,
            )
        ),
        "maximum_new_training_rows": int(
            cfg.get(
                "maximum_new_training_rows",
                1_000_000,
            )
        ),
        "drift": {
            "lookback_days": int(
                drift_cfg.get(
                    "lookback_days",
                    14,
                )
            ),
            "consecutive_windows": int(
                drift_cfg.get(
                    "consecutive_windows",
                    2,
                )
            ),
        },
        "performance": {
            "consecutive_windows": int(
                performance_cfg.get(
                    "consecutive_windows",
                    2,
                )
            ),
            "rmse_limit": float(
                performance_cfg.get(
                    "rmse_limit",
                    1375.0,
                )
            ),
            "mae_limit": float(
                performance_cfg.get(
                    "mae_limit",
                    990.0,
                )
            ),
            "absolute_bias_limit": float(
                performance_cfg.get(
                    "absolute_bias_limit",
                    900.0,
                )
            ),
        },
    }