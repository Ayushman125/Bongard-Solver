from src.quantifier_module import QuantifierModule, QuantifierType

def test_detect_quantifiers_universal_existential():
    qm = QuantifierModule(confidence_threshold=0.0)
    relationships = [
        {'predicate': 'touching'},
        {'predicate': 'touching'},
        {'predicate': 'touching'}
    ]
    objects = [{}, {}, {}]
    patterns = qm.detect_quantifiers(relationships, objects)
    assert any(p.quantifier_type == QuantifierType.UNIVERSAL for p in patterns)
    assert any(p.quantifier_type == QuantifierType.EXISTENTIAL for p in patterns)
