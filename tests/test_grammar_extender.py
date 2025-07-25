from src.grammar_extender import GrammarExtender

def test_operator_proposal():
    extender = GrammarExtender()
    out = extender.propose(['mirror'], {'context': 'symmetry'}, topk=1)
    assert isinstance(out, list) and out and isinstance(out[0], str)
    assert all(op not in ['mirror'] for op in out)

def test_propose_returns_candidates():
    extender = GrammarExtender()
    known_ops = ['mirror']
    proposals = extender.propose(known_ops, {'keywords': ['symmetry']}, topk=2)
    assert isinstance(proposals, list)
    assert all(op not in known_ops for op in proposals)
