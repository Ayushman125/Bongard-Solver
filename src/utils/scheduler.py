"""
AdaptiveScheduler: simple task batching for CPU vs GPU workloads.
Version: 0.1.0
"""

__version__ = "0.1.0"

from collections import deque

class AdaptiveScheduler:
    def __init__(self, cpu_workers: int = 4, gpu_workers: int = 1):
        self.cpu_queue = deque()
        self.gpu_queue = deque()

    def submit(self, fn, device: str = "cpu"):
        """
        Add a task to the CPU or GPU queue.
        Args:
            fn: Callable to execute.
            device: "cpu" or "gpu"
        """
        if device == "gpu":
            self.gpu_queue.append(fn)
        else:
            self.cpu_queue.append(fn)

    def run_all(self):
        """Execute all queued tasks, preferring CPU first."""
        while self.cpu_queue:
            self.cpu_queue.popleft()()
        while self.gpu_queue:
            self.gpu_queue.popleft()()
