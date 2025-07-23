import time
from integration.task_profiler import TaskProfiler

class AnytimeInference:
    def __init__(self, time_budget_ms: int = 200):
        self.profiler = TaskProfiler()
        self.budgets = {'coarse': 50, 'refine': 150, 'full': time_budget_ms}

    def ground(self, inputs: dict) -> dict:
        start = time.perf_counter() * 1000
        result = {}
        for stage, budget in self.budgets.items():
            with self.profiler.profile(f'ground_{stage}'):
                result[stage] = f"{stage}_output"
            elapsed = time.perf_counter()*1000 - start
            if elapsed > budget:
                break
        return result
