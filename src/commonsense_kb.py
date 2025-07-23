import json
import numpy as np

class CommonsenseKB:
    """
    Loads ConceptNet-lite JSON and provides predicate-based lookups.
    Expect file `data/conceptnet_lite.json` with entries:
      { predicate: { "head": [...], "tail": [...], "weight": ... } }
    """
    def __init__(self, path='data/conceptnet_lite.json'):
        with open(path, 'r') as f:
            self.cn = json.load(f)

    def query(self, predicate: str, context: list) -> dict:
        """
        Returns top–K entries under `predicate` for given context vector.
        """
        if predicate not in self.cn:
            return {}
        entries = self.cn[predicate]
        # rudimentary: return full list for now
        return entries
