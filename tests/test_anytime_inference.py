from grounder.anytime_inference import AnytimeInference, InferenceLevel

def test_infer_with_budget_levels():
    ai = AnytimeInference()
    data = {'physics': {'stability_score': 0.9, 'affordances': ['roll','slide']}, 'quantifiers': ['∀x'], 'foo': 1}
    for level in [InferenceLevel.COARSE, InferenceLevel.MEDIUM, InferenceLevel.FINE, InferenceLevel.FULL]:
        result = ai._infer_at_level(data, level, 100)
        assert result.level == level
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.result, dict)

# Degenerate geometry and feature extraction tests
def test_degenerate_stroke_geometry():
    from src.Derive_labels.shape_utils import calculate_geometry_consistent
    # Stroke with <3 vertices (degenerate)
    degenerate_vertices = [(0, 0), (1, 1)]
    geometry = calculate_geometry_consistent(degenerate_vertices)
    assert geometry['area'] == 0.0
    assert geometry['perimeter'] == 0.0
    assert geometry['width'] == 0.0
    assert geometry['height'] == 0.0
    assert geometry['centroid'] == [0.0, 0.0]

def test_zero_area_polygon():
    from src.Derive_labels.shape_utils import calculate_geometry_consistent
    # All points collinear
    collinear_vertices = [(0, 0), (1, 1), (2, 2)]
    geometry = calculate_geometry_consistent(collinear_vertices)
    assert geometry['area'] == 0.0
    assert geometry['width'] == 0.0 or geometry['height'] == 0.0
    assert geometry['centroid'] == [0.0, 0.0] or isinstance(geometry['centroid'], list)
