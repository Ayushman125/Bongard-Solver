from src.cross_domain_reasoner import CrossDomainReasoner, ReasoningMode

def test_select_reasoning_mode_fusion():
    reasoner = CrossDomainReasoner()
    phys = {'stability_score': 0.5}
    viz = {}
    ctx = 'why is this stable?'
    mode = reasoner._select_reasoning_mode(phys, viz, ctx)
    assert mode == ReasoningMode.FUSION

def test_select_reasoning_mode_physics_only():
    reasoner = CrossDomainReasoner()
    phys = {'stability_score': 0.9}
    viz = {}
    ctx = 'object'
    mode = reasoner._select_reasoning_mode(phys, viz, ctx)
    assert mode == ReasoningMode.PHYSICS_ONLY

# Degenerate feature reasoning test
def test_reasoning_with_degenerate_features():
    reasoner = CrossDomainReasoner()
    # All features degenerate (zero/empty/default geometry)
    degenerate_phys = {'stability_score': 0.0, 'area': 0.0, 'curvature': 0.0}
    degenerate_viz = {'geometry': {'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'width': 0.0, 'height': 0.0}}
    ctx = 'degenerate test'
    mode = reasoner._select_reasoning_mode(degenerate_phys, degenerate_viz, ctx)
    assert mode in [ReasoningMode.FUSION, ReasoningMode.PHYSICS_ONLY, ReasoningMode.VISUAL_ONLY]
