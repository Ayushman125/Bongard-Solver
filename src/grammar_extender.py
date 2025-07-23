import yaml
from integration.task_profiler import TaskProfiler

class GrammarExtender:
    """
    Stub for dynamic DSL evolution:
      - measure coverage
      - propose new operators
    """
    def __init__(self, grammar_path='config/grammar.yaml'):
        self.profiler = TaskProfiler()
        self.grammar = yaml.safe_load(open(grammar_path))

    def propose(self, coverage: float) -> list:
        """
        If coverage < 0.8, propose 'diff_ratio' operator stub.
        """
        with self.profiler.profile('grammar_extension'):
            if coverage < 0.8:
                return [{'op': 'diff_ratio', 'params': {'alpha': 0.1}}]
            return []
