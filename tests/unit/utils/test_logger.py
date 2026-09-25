import logging

from mlops_sales_forecasting.utils.logger import get_logger


def test_get_logger_returns_configured_logger() -> None:
    logger = get_logger("test.configured")

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_get_logger_does_not_add_duplicate_handlers() -> None:
    first_logger = get_logger("test.duplicate")
    second_logger = get_logger("test.duplicate")

    assert first_logger is second_logger
    assert len(second_logger.handlers) == 1


def test_get_logger_accepts_custom_level() -> None:
    logger = get_logger("test.level", level=logging.DEBUG)

    assert logger.level == logging.DEBUG