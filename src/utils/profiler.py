"""
TaskProfiler: decorator to log function latency into JSON events.
Version: 0.1.0
"""

__version__ = "0.1.0"

import time, json, threading
from pathlib import Path

_LOG_PATH = Path("logs/profiler_events.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True)

_lock = threading.Lock()

def profile(module: str):
    """
    Decorator to measure execution time and log as JSON per call.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            event = {
                "module": module,
                "timestamp": time.time(),
                "latency_ms": latency_ms
            }
            with _lock, open(_LOG_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")
            return result
        return wrapper
    return decorator
