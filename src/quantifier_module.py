import numpy as np

class QuantifierModule:
    """
    Detects universal/existential patterns in a set of grounded relations.
    If every relation holds → ∀; if at least one → ∃.
    """
    def detect(self, relations: list[bool]) -> dict:
        return {
            'forall': all(relations),
            'exists': any(relations)
        }
