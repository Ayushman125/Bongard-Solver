import numpy as np
from typing import Sequence

class QuantifierModule:
    """
    Detects universal/existential patterns in a set of grounded relations.
    If every relation holds → ∀; if at least one → ∃.
    """
    def detect(self, relations: Sequence[bool]) -> dict:
        return {
            'forall': all(relations),
            'exists': any(relations)
        }
