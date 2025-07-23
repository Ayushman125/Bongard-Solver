from src.grammar_extender import GrammarExtender

def test_operator_proposal():
    extender = GrammarExtender()
    out = extender.propose(['mirror'], {'context': 'symmetry'}, topk=1)
    assert isinstance(out, list) and out and isinstance(out[0], str)
    assert all(op not in ['mirror'] for op in out)
