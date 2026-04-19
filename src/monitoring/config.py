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