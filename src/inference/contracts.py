from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class InferenceArtifacts:
    """
    Runtime artifacts required for inference.
    Kept generic so different providers can consume different assets.
    """
    assets: dict[str, Any] = field(default_factory=dict)

    def require(self, key: str) -> Any:
        if key not in self.assets:
            raise ValueError(f"Missing required inference artifact: '{key}'")
        return self.assets[key]


@dataclass
class InferenceContext:
    """
    Optional runtime context for provider-specific resolution.
    """
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class InferenceBuildRequest:
    """
    Generic internal contract for feature building.
    """
    validated_df: pd.DataFrame
    config: dict[str, Any]
    artifacts: InferenceArtifacts
    context: InferenceContext = field(default_factory=InferenceContext)