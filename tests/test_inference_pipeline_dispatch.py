import pandas as pd
import pytest

from src.inference.contracts import InferenceArtifacts, InferenceContext
from src.inference.pipeline import build_inference_features


def test_build_inference_features_rejects_unknown_problem_type():
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        build_inference_features(
            validated_df=pd.DataFrame([{"x": 1}]),
            config={"project": {"problem_type": "unknown"}},
            artifacts=InferenceArtifacts(assets={}),
            context=InferenceContext(values={}),
        )