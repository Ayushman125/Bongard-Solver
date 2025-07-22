import time, json, threading
from pathlib import Path

_LOG = Path("logs/profiler.jsonl")
_LOG.parent.mkdir(exist_ok=True)
_lock = threading.Lock()

def profile(module: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            out = fn(*args, **kwargs)
            latency = (time.time() - t0) * 1000
            event = {
              "module": module,
              "timestamp": time.time(),
              "latency_ms": latency
            }
            with _lock, _LOG.open("a") as f:
                f.write(json.dumps(event) + "\n")
            return out
        return wrapper
    return decorator
