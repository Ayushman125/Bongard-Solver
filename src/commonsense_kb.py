import json
from integration.data_validator import DataValidator

class KBLoadError(Exception):
    pass

class CommonsenseKB:
    """
    Loads ConceptNet-lite JSON and provides predicate-based lookups.
    Expect file `data/conceptnet_lite.json` with entries:
      { predicate: { "head": [...], "tail": [...], "weight": ... } }
    """
    def __init__(self, path: str = 'data/conceptnet_lite.json'):
        self.dv = DataValidator()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.dv.validate(data, 'conceptnet_lite.schema.json')
            self.cn = data
        except Exception as e:
            raise KBLoadError(f"Failed to load KB: {e}")

    def query(self, predicate: str, context: list) -> dict:
        """
        Returns top–K entries under `predicate` for given context vector.
        """
        if predicate not in self.cn:
            return {}
        entries = self.cn[predicate]
        # rudimentary: return full list for now
        return entries
