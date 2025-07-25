
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

# Legacy compatibility: TaskProfiler class
class TaskProfiler:
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
