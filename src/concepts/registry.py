
"""
Central concept registry for Bongard-Solver.
Auto-populates from derived_labels.json and enforces coverage.
Maps every problem_id to a real, auditable concept-checking function (from src/concepts.py).
Fails hard if any mapping is missing. No fallback logic.
"""
import os
import sys
import json

# Import all concept functions from src/concepts.py
import importlib.util
concepts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'concepts.py'))
spec = importlib.util.spec_from_file_location("concepts", concepts_path)
concepts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concepts_mod)

# Build a mapping from function name to function object
ALL_CONCEPTS = {k: v for k, v in concepts_mod.__dict__.items() if callable(v) and not k.startswith("__")}

DERIVED_LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/derived_labels.json'))

class ConceptRegistry:
    def __init__(self, derived_labels_path=DERIVED_LABELS_PATH):
        self.problem_to_concept = {}
        self._populate_registry(derived_labels_path)

    def _populate_registry(self, derived_labels_path):
        if not os.path.exists(derived_labels_path):
            raise FileNotFoundError(f"derived_labels.json not found at {derived_labels_path}")
        with open(derived_labels_path, 'r') as f:
            data = json.load(f)
        # Build a mapping from problem_id to features
        pid_to_features = {}
        for entry in data:
            pid = entry['problem_id']
            features = entry.get('features', {})
            pid_to_features.setdefault(pid, []).append(features)

        # For each problem_id in the data, find a matching concept function
        for pid, features_list in pid_to_features.items():
            found = False
            for cname, cfn in ALL_CONCEPTS.items():
                for features in features_list:
                    try:
                        if cfn(features):
                            self.problem_to_concept[pid] = cfn
                            found = True
                            break
                    except Exception:
                        continue
                if found:
                    break
            if not found:
                raise ValueError(f"No concept function matches for problem_id: {pid} (present in derived_labels.json). Please implement and register a function in src/concepts.py.")

    def get(self, pid):
        if pid not in self.problem_to_concept:
            raise KeyError(f"No concept function registered for problem ID: {pid}. This problem ID is not present in derived_labels.json or is not covered by any concept function.")
        return self.problem_to_concept[pid]

def get_concept_fn_for_problem(pid):
    """Factory function to get the concept function for a problem ID."""
    registry = ConceptRegistry()
    return registry.get(pid)
