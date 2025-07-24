class NoPredicateFound(Exception):
    pass


"""
Self-maintaining concept registry for Bongard-Solver.
Auto-induces and caches concept predicates for every problem in derived_labels.json.
No manual wiring required; new problems are handled automatically.
"""


import os
import sys
import json
import time
from pathlib import Path
import yaml
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from .auto_inducer import induce
from .auto_tree import induce_tree

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

def run_with_timeout(fn, args=(), timeout=2):
    with ProcessPoolExecutor(1) as exe:
        fut = exe.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeoutError:
            fut.cancel()
            raise TimeoutError(f"{fn.__name__} timed out after {timeout}s")

def _spec_to_lambda(spec):
    # Convert a spec dict to a Python lambda for runtime use
    sig = spec['signature']
    param = spec['param']
    typ = spec.get('type','')
    feats = spec.get('features',[])
    # Single-feature bool
    if typ == 'bool':
        feat = feats[0]
        return lambda f: bool(f.get(feat, False)) == bool(param)
    # Single-feature threshold
    if typ == 'threshold':
        feat = feats[0]
        if '>' in sig:
            return lambda f: f.get(feat, 0) > param
        else:
            return lambda f: f.get(feat, 0) < param
    # Single-feature range
    if typ == 'range':
        feat = feats[0]
        lo, hi = param
        return lambda f: lo <= f.get(feat, 0) <= hi
    # Two-feature AND (bool)
    if typ == 'and_bool':
        f1, f2 = feats
        v1, v2 = param
        return lambda f: (f.get(f1, False)==v1) and (f.get(f2, False)==v2)
    # Two-feature AND (range)
    if typ == 'and_range':
        (min1,max1),(min2,max2) = param
        f1, f2 = feats
        return lambda f: min1 <= f.get(f1, 1e9) <= max1 and min2 <= f.get(f2, 1e9) <= max2
    # fallback: always False
    return lambda f: False

class ConceptRegistry:
    _singleton = None

    def __new__(cls, derived_labels_path=DERIVED_LABELS_PATH, cache_path=CACHE_PATH):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self, derived_labels_path=DERIVED_LABELS_PATH, cache_path=CACHE_PATH):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.cache_path = Path(cache_path)
        self.funcs = {}
        self.specs = {}
        self.pid_to_samples = _load_derived_labels(derived_labels_path)
        self._load_cache()
        self._initialized = True

    def _load_cache(self):
        if self.cache_path.exists():
            self.specs = yaml.safe_load(self.cache_path.read_text()) or {}
        else:
            self.specs = {}
        # Build runtime lambdas
        for pid, spec in self.specs.items():
            self.funcs[pid] = _spec_to_lambda(spec)

    def _factory_program_membership(self, pid, positives):
        pos_seqs = {tuple(p["action_program"]) for p in positives if "action_program" in p}
        return lambda features, seqs=pos_seqs: tuple(features.get("action_program", [])) in seqs

    def _induce_for(self, pid):
        import logging
        samples = self.pid_to_samples.get(pid, [])
        positives = [f for f, y in samples if y == 1]
        negatives = [f for f, y in samples if y == 0]
        if not positives or not negatives:
            logging.warning(f"ConceptRegistry: degenerate problem {pid}, using fallback.")
            fn = self._factory_program_membership(pid, positives)
            self.specs[pid] = {'signature': 'program_membership', 'param': None, 'features': [], 'type': 'membership'}
            self.funcs[pid] = fn
            return fn
        fn = None
        try:
            logging.info(f"ConceptRegistry: inducing for problem '{pid}'")
            start = time.perf_counter()
            try:
                spec = run_with_timeout(induce, (pid, positives, negatives), timeout=2)
                self.specs[pid] = spec
                fn = _spec_to_lambda(spec)
                logging.info(f"INFO  Auto-derived concept for {pid} → {spec['signature']}")
            except TimeoutError:
                logging.error(f"induce() timed out for {pid}, falling back")
                raise NoPredicateFound()
        except NoPredicateFound:
            logging.warning(f"No predicate found for {pid}, using program membership fallback.")
            fn = self._factory_program_membership(pid, positives)
            self.specs[pid] = {'signature': 'program_membership', 'param': None, 'features': [], 'type': 'membership'}
        except Exception as e:
            logging.warning(f"Template induction failed for {pid} ({e}), using decision tree.")
            try:
                fn = run_with_timeout(induce_tree, (pid,), timeout=1)
                self.specs[pid] = {'signature': 'decision_tree', 'param': None, 'features': [], 'type': 'tree'}
            except Exception as e2:
                logging.error(f"induce_tree() failed for {pid}: {e2}")
                fn = self._factory_program_membership(pid, positives)
                self.specs[pid] = {'signature': 'program_membership', 'param': None, 'features': [], 'type': 'membership'}
        if fn is None:
            logging.warning(f"All induction failed for {pid}, using program membership as last resort.")
            fn = self._factory_program_membership(pid, positives)
            self.specs[pid] = {'signature': 'program_membership', 'param': None, 'features': [], 'type': 'membership'}
        self.funcs[pid] = fn
        self.cache_path.write_text(yaml.safe_dump(self.specs))
        logging.info(f"ConceptRegistry: done '{pid}'")
        return fn

    def get(self, pid):
        fn = self.funcs.get(pid)
        if fn is None:
            # Lazy induction for new problems
            fn = self._induce_for(pid)
        return fn

_GLOBAL_REGISTRY = ConceptRegistry()

def get_concept_fn_for_problem(pid):
    """Factory function to get the concept function for a problem ID."""
    fn = _GLOBAL_REGISTRY.get(pid)
    if fn is None:
        raise KeyError(f"Concept missing for '{pid}' — registry keys: {list(_GLOBAL_REGISTRY.funcs.keys())}")
    return fn
