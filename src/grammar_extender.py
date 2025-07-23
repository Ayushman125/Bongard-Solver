import yaml
from integration.task_profiler import TaskProfiler
from integration.data_validator import DataValidator

class GrammarExtender:
    """
    Stub for dynamic DSL evolution:
      - measure coverage
      - propose new operators
    """
    def __init__(self, grammar_path: str = 'config/grammar.yaml'):
        self.profiler = TaskProfiler()
        self.dv = DataValidator()
        try:
            cfg = yaml.safe_load(open(grammar_path))
            self.dv.validate(cfg, 'grammar_config.schema.json')
            self.grammar = cfg
        except Exception as e:
            raise RuntimeError(f"Failed loading grammar: {e}")

    def propose(self, coverage: float, tau: float = 0.3) -> list:
        """
        If coverage < 0.8, propose 'diff_ratio' operator stub.
        """
        with self.profiler.profile('grammar_extension'):
            if coverage < 0.8:
                return [{'op': 'diff_ratio', 'params': {'alpha': 0.1, 'tau': tau}}]
            return []
