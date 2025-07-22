import json
import pytest
from src.utils.commonsense_kb import CommonsenseKB

@pytest.fixture(autouse=True)
def sample_kb(tmp_path):
    data = [["dog","wants","ball"],["cat","avoids","water"]]
    f = tmp_path / "kb.json"
    f.write_text(json.dumps(data))
    return str(f)

def test_query(sample_kb):
    kb = CommonsenseKB(sample_kb)
    assert ("dog","ball") in kb.query("wants")
    assert ("cat","water") in kb.query("avoids")
