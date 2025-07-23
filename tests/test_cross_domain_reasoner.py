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
