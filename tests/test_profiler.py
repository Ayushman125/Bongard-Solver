import os, json
from src.utils.profiler import profile

LOG = "logs/profiler.jsonl"

@profile("test_mod")
def dummy():
    return 42

def test_profiler_creates_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    if os.path.exists(LOG):
        os.remove(LOG)
    result = dummy()
    assert result == 42
    with open(LOG) as f:
        line = json.loads(f.readline())
    assert line["module"] == "test_mod"
    assert "latency_ms" in line
