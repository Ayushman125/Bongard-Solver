"""
CommonsenseKB: loader for ConceptNet-lite and simple predicate queries.
Version: 0.1.0
"""

__version__ = "0.1.0"

import json
from collections import defaultdict
from typing import List, Tuple

class CommonsenseKB:
    def __init__(self, path: str):
        """
        Args:
            path: File path to trimmed ConceptNet-lite JSON.
        """
        with open(path, "r") as f:
            edges = json.load(f)
        # Build predicate→objects index
        self._index = defaultdict(list)
        for subj, pred, obj in edges:
            self._index[pred].append((subj, obj))

    def query(self, predicate: str) -> List[Tuple[str, str]]:
        """
        Retrieve subject-object pairs for a given predicate.
        E.g. query("wants") => [("dog", "ball"), ...]
        """
        return self._index.get(predicate, [])
