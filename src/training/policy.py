def should_skip_training(drift_detected: bool, force_run: bool) -> bool:
    """
    Returns True when the pipeline should skip retraining
    and only evaluate the current champion.
    """
    return not drift_detected and not force_run


def get_run_strategy(drift_detected: bool, force_run: bool) -> str:
    """
    Returns the pipeline strategy label.
    """
    if drift_detected:
        return "EMERGENCY"
    if force_run:
        return "FORCED"
    return "STABLE"


def should_refresh_api(new_champion_crowned: bool) -> bool:
    """
    Returns True if the API should be refreshed after model promotion.
    """
    return new_champion_crowned