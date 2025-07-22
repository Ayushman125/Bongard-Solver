__version__ = "0.1.0"
"""
AdaptiveScheduler

Simple CPU/GPU task queue. Ensures CPU tasks drain before GPU:

  sched = AdaptiveScheduler()
  sched.submit(fn, device="cpu"|"gpu")
  sched.run_all()
"""
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
            self.gpu_q.append(fn)
        else:
            self.cpu_q.append(fn)

    def run_all(self):
        """Execute all queued tasks, preferring CPU first."""
        while self.cpu_q:
            self.cpu_q.popleft()()
        while self.gpu_q:
            self.gpu_q.popleft()()
