
"""
Task Profiler
Version: 0.1.0

This module provides a simple decorator for profiling the latency of functions.
Also provides a TaskProfiler class for compatibility with legacy imports.
"""

__version__ = "0.1.0"

import time
from functools import wraps

def profile(func):
    """
    A decorator that profiles the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

# --- Add start_task and end_task for compatibility ---
_TASK_START_TIMES = {}
def start_task(name):
    """Mark the start of a profiling task."""
    _TASK_START_TIMES[name] = time.time()
    print(f"[Profiler] Task '{name}' started.")

def end_task(name):
    """Mark the end of a profiling task and print elapsed time."""
    start = _TASK_START_TIMES.pop(name, None)
    if start is not None:
        elapsed = time.time() - start
        print(f"[Profiler] Task '{name}' ended. Elapsed: {elapsed:.4f}s")
    else:
        print(f"[Profiler] Task '{name}' end called without start.")

# Legacy compatibility: TaskProfiler class
class TaskProfiler:
    def log_latency(self, name, latency, metadata=None):
        """Logs latency for a given operation. Prints to stdout for now."""
        print(f"[Profiler] {name}: {latency:.2f} ms", end="")
        if metadata:
            print(f" | Metadata: {metadata}")
        else:
            print()
    """
    Class-based profiler for function execution time. Use as a decorator or context manager.
    """
    def __init__(self, func=None):
        self.func = func
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{self.func.__name__}' executed in {end_time - start_time:.4f}s")
        return result

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        print(f"TaskProfiler context executed in {elapsed:.4f}s")

    class _ProfileContext:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self._start = time.time()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self._start
            print(f"Profile '{self.name}' executed in {elapsed:.4f}s")

    def profile(self, name):
        """Returns a context manager for profiling a code block with a given name."""
        return self._ProfileContext(name)
