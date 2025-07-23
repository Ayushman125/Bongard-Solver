from grounder.anytime_inference import AnytimeInference, InferenceLevel

def test_infer_with_budget_levels():
    ai = AnytimeInference()
    data = {'physics': {'stability_score': 0.9, 'affordances': ['roll','slide']}, 'quantifiers': ['∀x'], 'foo': 1}
    for level in [InferenceLevel.COARSE, InferenceLevel.MEDIUM, InferenceLevel.FINE, InferenceLevel.FULL]:
        result = ai._infer_at_level(data, level, 100)
        assert result.level == level
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.result, dict)
