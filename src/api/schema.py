from typing import Any

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    """
    Generic prediction request contract for blueprint-ready APIs.
    Supports single-row and multi-row inference via a list of input records.
    """
    inputs: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of input records for inference",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata, routing hints or execution context",
    )

    @model_validator(mode="after")
    def validate_inputs_not_empty(self):
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        return self


class PredictionResponse(BaseModel):
    """
    Generic prediction response contract.
    """
    predictions: list[float]
    status: str = "success"
    metadata: dict[str, Any] | None = None