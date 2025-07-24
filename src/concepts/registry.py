

"""
Self-maintaining concept registry for Bongard-Solver.
Auto-induces and caches concept predicates for every problem in derived_labels.json.
No manual wiring required; new problems are handled automatically.
"""
import os
import sys
import json
from pathlib import Path
import yaml
from .auto_inducer import induce

DERIVED_LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/derived_labels.json'))
CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/concepts_auto.yaml'))

def _load_derived_labels(path=DERIVED_LABELS_PATH):
    with open(path, 'r') as f:
        data = json.load(f)
    # Map problem_id to list of (features, label)
    pid_to_samples = {}
    for entry in data:
        pid = entry['problem_id']
        # Normalize label: category_1 → 1, category_0 → 0, or fallback to int/str
        label = entry.get('label', None)
        if label in ('category_1', 1, 'positive', True):
            y = 1
        elif label in ('category_0', 0, 'negative', False):
            y = 0
        else:
            y = int(label) if isinstance(label, int) else 0
        features = entry.get('features', {})
        pid_to_samples.setdefault(pid, []).append((features, y))
    return pid_to_samples

def _spec_to_lambda(spec):
    # Convert a spec dict to a Python lambda for runtime use
    sig = spec['signature']
    param = spec['param']
    if sig.startswith('is_convex'):
        return lambda f: f.get('is_convex', False)
    if sig.startswith('num_straight=='):
        n = param
        return lambda f: f.get('num_straight', None) == n
    if sig.startswith('has_quadrangle'):
        return lambda f: f.get('has_quadrangle', False)
    if sig.startswith('nact_range'):
        lo, hi = param
        return lambda f: lo <= f.get('num_strokes', 0) <= hi
    if sig.startswith('symmetry_score<'):
        t = param
        return lambda f: f.get('symmetry_score', 1e9) < t
    # fallback: always False
    return lambda f: False

class ConceptRegistry:
    def __init__(self, derived_labels_path=DERIVED_LABELS_PATH, cache_path=CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.funcs = {}
        self.specs = {}
        self._load_cache()
        self._scan_and_update(derived_labels_path)

    def _load_cache(self):
        if self.cache_path.exists():
            self.specs = yaml.safe_load(self.cache_path.read_text()) or {}
        else:
            self.specs = {}
        # Build runtime lambdas
        for pid, spec in self.specs.items():
            self.funcs[pid] = _spec_to_lambda(spec)

    def _scan_and_update(self, derived_labels_path):
        pid_to_samples = _load_derived_labels(derived_labels_path)
        updated = False
        for pid, samples in pid_to_samples.items():
            if pid in self.funcs:
                continue
            positives = [f for f, y in samples if y == 1]
            negatives = [f for f, y in samples if y == 0]
            if not positives or not negatives:
                continue  # skip degenerate
            spec = induce(pid, positives, negatives)
            self.specs[pid] = spec
            self.funcs[pid] = _spec_to_lambda(spec)
            print(f"INFO  Auto-derived concept for {pid} → {spec['signature']}")
            updated = True
        if updated:
            self.cache_path.write_text(yaml.safe_dump(self.specs))

    def get(self, pid):
        if pid not in self.funcs:
            raise KeyError(f"No concept function registered for problem ID: {pid}. This problem ID is not present in derived_labels.json or could not be induced.")
        return self.funcs[pid]

def get_concept_fn_for_problem(pid):
    """Factory function to get the concept function for a problem ID."""
    registry = ConceptRegistry()
    return registry.get(pid)
