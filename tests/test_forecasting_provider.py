import pandas as pd

from src.inference.contracts import InferenceArtifacts, InferenceContext, InferenceBuildRequest
from src.inference.forecasting_provider import build_forecasting_inference_features


def test_build_forecasting_features_requires_store_column_or_context():
    validated_df = pd.DataFrame([{"Promo": 1}])

    request = InferenceBuildRequest(
        validated_df=validated_df,
        config={"project": {"problem_type": "forecasting"}},
        artifacts=InferenceArtifacts(
            assets={"store_metadata": pd.DataFrame(), "store_state": {}}
        ),
        context=InferenceContext(values={}),
    )

    try:
        build_forecasting_inference_features(request)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Store" in str(e)