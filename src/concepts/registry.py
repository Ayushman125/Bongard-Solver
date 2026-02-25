

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
from .exceptions import NoPredicateFound

DERIVED_LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/derived_labels.json'))
CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/concepts_auto.yaml'))

import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def _flatten_features(features, prioritized=None):
    # Flatten features dict to vector, using prioritized order if given
    # If a feature is a dict (e.g., ngram, stroke_type_distribution), flatten its values (sorted by key)

    def to_numeric(x):
        # Convert to float if possible, else 0.0
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        try:
            return float(x)
        except Exception:
            return 0.0

    # For dict-valued features, we need a global key order for each prioritized feature
    # Build a map: feature_name -> sorted list of all possible subkeys seen in any sample
    # We'll cache this in a static variable on the function
    if not hasattr(_flatten_features, '_dict_key_order'):
        _flatten_features._dict_key_order = {}

    dict_key_order = _flatten_features._dict_key_order

    # If this is the first call in a batch, build the dict_key_order for all prioritized features
    if prioritized is not None and not dict_key_order:
        # This is a hack: we expect the caller to call _flatten_features on all samples in a batch before using the vectors
        # So we can build the key order by scanning all features in the batch
        import inspect
        frame = inspect.currentframe().f_back
        # Find the list comprehension in _feature_diversity or _scan_and_update
        local_vars = frame.f_locals
        # Try to find the list of all features being processed
        for v in local_vars.values():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                for k in prioritized:
                    all_keys = set()
                    for f in v:
                        val = f.get(k, None)
                        if isinstance(val, dict):
                            all_keys.update(val.keys())
                    if all_keys:
                        dict_key_order[k] = sorted(all_keys)

    flat = []
    for k in prioritized if prioritized is not None else sorted(features.keys()):
        v = features.get(k, 0.0)
        if isinstance(v, dict):
            # Use global key order for this feature
            key_order = dict_key_order.get(k, [])
            for subk in key_order:
                flat.append(to_numeric(v.get(subk, 0.0)))
        elif isinstance(v, (list, tuple)):
            # Always pad/truncate to length of first seen list for this feature
            # For simplicity, just use up to 10 elements
            vals = [to_numeric(x) for x in v]
            flat.extend(vals[:10] + [0.0]*(10-len(vals)))
        else:
            flat.append(to_numeric(v))
    return flat

def _cosine_sim(a, b):
    # Compute cosine similarity between two feature dicts
    a_vec = np.array(a).reshape(1, -1)
    b_vec = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a_vec, b_vec)[0, 0])

def _feature_diversity(samples, prioritized):
    # Compute mean pairwise cosine distance (1-similarity) for a set of feature dicts
    if len(samples) < 2:
        return 0.0
    vecs = [_flatten_features(f, prioritized) for f in samples]
    sims = [ _cosine_sim(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs)) ]
    return float(np.mean([1-s for s in sims]))

def _load_derived_labels(path=DERIVED_LABELS_PATH, prioritized=None):
    import logging
    with open(path, 'r') as f:
        data = json.load(f)
    # Map problem_id to list of (features, label)
    pid_to_samples = {}
    for i, entry in enumerate(data):
        pid = entry.get('problem_id', None)
        if pid is None:
            logging.warning(f"Sample {i} is missing 'problem_id'. Skipping sample.")
            continue
        # Normalize label: category_1 → 1, category_0 → 0, or fallback to int/str
        label = entry.get('label', None)
        if label in ('category_1', 1, 'positive', True):
            y = 1
        elif label in ('category_0', 0, 'negative', False):
            y = 0
        else:
            y = int(label) if isinstance(label, int) else 0
        # Aggregate all relevant features: per-stroke, image, physics, composition, stroke_type, relational, sequential, topological
        features = {}
        # Defensive: skip sample if it has no strokes and no feature dicts
        has_any_features = False
        for stroke in entry.get('strokes', []):
            if 'specific_features' in stroke:
                features.update(stroke.get('specific_features', {}))
                has_any_features = True
        for key in ["image_features", "physics_features", "composition_features", "stroke_type_features",
                    "relational_features", "sequential_features", "topological_features"]:
            comp = entry.get(key, {})
            if isinstance(comp, dict) and comp:
                features.update(comp)
                has_any_features = True
        if not has_any_features:
            logging.warning(f"Sample for problem {pid} is missing all expected feature keys. Skipping sample.")
            continue
        pid_to_samples.setdefault(pid, []).append((features, y))
        # Debug: print the first 3 feature dicts for inspection
        if i < 3:
            print(f"DEBUG registry.py: Sample {i} for problem {pid} label={y}\nFeatures: {json.dumps(features, indent=2)}\n")
    return pid_to_samples

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
        prioritized_features = [
            'geometric_complexity', 'homogeneity_score', 'shape_diversity', 'pattern_regularity',
            'dominant_stroke_type', 'dominant_shape_modifier', 'visual_complexity', 'irregularity_score',
            'num_straight_segments', 'num_arcs', 'has_obtuse_angle', 'has_quadrangle', 'compactness',
            'eccentricity', 'aspect_ratio', 'area', 'perimeter', 'is_convex', 'symmetry_score', 'rotational_symmetry',
            # Add relational, sequential, topological features
            'intersections', 'adjacency', 'containment', 'overlap',
            'ngram', 'alternation', 'regularity', 'num_strokes'
        ]
        pid_to_samples = _load_derived_labels(derived_labels_path, prioritized=prioritized_features)
        updated = False
        for pid, samples in pid_to_samples.items():
            if pid in self.funcs:
                continue
            positives = [f for f, y in samples if y == 1]
            negatives = [f for f, y in samples if y == 0]
            if not positives or not negatives:
                continue  # skip degenerate
            # --- Hard Negative Mining: select negatives with max feature diversity from positives ---
            # Compute mean feature vector for positives
            pos_vecs = [_flatten_features(f, prioritized_features) for f in positives]
            neg_vecs = [_flatten_features(f, prioritized_features) for f in negatives]
            pos_mean = np.mean(pos_vecs, axis=0)
            # Compute cosine similarity of each negative to positive mean
            neg_sims = [ _cosine_sim(nv, pos_mean) for nv in neg_vecs ]
            # Select negatives with lowest similarity (most different)
            neg_sorted = [negatives[i] for i in np.argsort(neg_sims)]
            # Validate feature diversity among selected negatives
            best_negatives = []
            for n in neg_sorted:
                if not best_negatives:
                    best_negatives.append(n)
                else:
                    # Only add if mean pairwise diversity remains above threshold
                    test = best_negatives + [n]
                    div = _feature_diversity(test, prioritized_features)
                    if div > 0.05:  # threshold for diversity
                        best_negatives.append(n)
                if len(best_negatives) >= min(10, len(negatives)):
                    break
            # Use best_negatives for induction
            try:
                spec = induce(pid, positives, best_negatives)
                self.specs[pid] = spec
                self.funcs[pid] = _spec_to_lambda(spec)
                print(f"INFO  Auto-derived concept for {pid} → {spec['signature']}")
                updated = True
            except NoPredicateFound as e:
                # Fallback: try prioritized single features
                fallback_found = False
                for feat in prioritized_features:
                    pos_vals = [f.get(feat, None) for f in positives]
                    neg_vals = [f.get(feat, None) for f in best_negatives]
                    if (all(v is not None for v in pos_vals + neg_vals)
                        and (len(set(pos_vals)) > 1 or len(set(neg_vals)) > 1 or set(pos_vals) != set(neg_vals))):
                        # Try induction with only this feature
                        pos_feat = [{feat: f[feat]} for f in positives if feat in f]
                        neg_feat = [{feat: f[feat]} for f in best_negatives if feat in f]
                        try:
                            spec = induce(pid, pos_feat, neg_feat)
                            self.specs[pid] = spec
                            self.funcs[pid] = _spec_to_lambda(spec)
                            print(f"INFO  Fallback concept for {pid} using '{feat}' → {spec['signature']}")
                            updated = True
                            fallback_found = True
                            break
                        except NoPredicateFound:
                            continue
                if not fallback_found:
                    print(f"WARNING: {e}. Skipping problem {pid}.")
                    continue
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
