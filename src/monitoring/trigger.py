from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.monitoring.retraining_policy import (
    RetrainingDecision,
    decide_retraining,
)
from src.monitoring.signal_collector import (
    collect_retraining_signals,
)


def evaluate_retraining(
    *,
    evaluated_at: datetime
    | pd.Timestamp
    | None = None,
) -> RetrainingDecision:
    """
    Collect evidence and evaluate the central retraining policy.

    This function has no side effects:
    it neither trains a model nor changes retraining state.
    """

    signals = collect_retraining_signals(
        evaluated_at=evaluated_at
    )

    return decide_retraining(signals)