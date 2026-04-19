from src.inference.contracts import InferenceArtifacts, InferenceContext


def test_inference_artifacts_require_success():
    artifacts = InferenceArtifacts(assets={"store_metadata": "x"})
    assert artifacts.require("store_metadata") == "x"


def test_inference_context_get_default():
    context = InferenceContext(values={})
    assert context.get("missing", "fallback") == "fallback"