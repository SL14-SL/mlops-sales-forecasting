from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Feature records submitted for time-series forecasting."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[dict[str, Any]] = Field(
        min_length=1,
    )


class PredictionResult(BaseModel):
    """Forecast produced for one row and horizon step."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0)
    horizon_step: int = Field(ge=1)
    prediction: float


class PredictionResponse(BaseModel):
    """Successful forecasting response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    release_id: str = Field(min_length=1)
    predictions: list[PredictionResult]
