from src.training.policy import should_skip_training, get_run_strategy, should_refresh_api


def test_should_skip_training_returns_true_when_stable_and_not_forced():
    assert should_skip_training(False, False) is True


def test_should_skip_training_returns_false_when_drift_detected():
    assert should_skip_training(True, False) is False


def test_should_skip_training_returns_false_when_force_run_enabled():
    assert should_skip_training(False, True) is False


def test_get_run_strategy_returns_emergency_for_drift():
    assert get_run_strategy(True, False) == "EMERGENCY"


def test_get_run_strategy_returns_forced_for_force_run():
    assert get_run_strategy(False, True) == "FORCED"


def test_get_run_strategy_returns_stable_otherwise():
    assert get_run_strategy(False, False) == "STABLE"


def test_should_refresh_api_returns_true_for_new_champion():
    assert should_refresh_api(True) is True


def test_should_refresh_api_returns_false_without_new_champion():
    assert should_refresh_api(False) is False