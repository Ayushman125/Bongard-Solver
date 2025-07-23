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
    def __init__(self):
        # match test names exactly
        self.cpu_q = deque()
        self.gpu_q = deque()

    def submit(self, fn, device: str = "cpu"):
        """
        Add a no-arg callable to the CPU or GPU queue.
        """
        if device == "gpu":
            self.gpu_q.append(fn)
        else:
            self.cpu_q.append(fn)

    def run_all(self):
        """
        Execute all CPU tasks first, then all GPU tasks.
        """
        while self.cpu_q:
            task = self.cpu_q.popleft()
            task()

        while self.gpu_q:
            task = self.gpu_q.popleft()
            task()
        while self.gpu_q:
            self.gpu_q.popleft()()
