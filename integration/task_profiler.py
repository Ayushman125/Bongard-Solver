"""
Task Profiler
Version: 0.1.0

This module provides a simple decorator for profiling the latency of functions.
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
