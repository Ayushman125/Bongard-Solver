import numpy as np
from itertools import combinations
from sklearn.metrics import mutual_info_score

DEFAULT_CANDIDATES = {
    "larger_than": 1.15,
    "aspect_sim": 0.20,
    "near": 32.0,
    "para": 12.0,
}

class PredicateMiner:
    def __init__(self, search_grid=None):
        self.grid = search_grid or DEFAULT_CANDIDATES.copy()

    def fit(self, objects):
        """Pick thresholds that maximise Δ = support_pos – support_neg."""
        pos = [o for o in objects if o["shape_label"] == "positive"]
        neg = [o for o in objects if o["shape_label"] == "negative"]
        best = {}
        for name, base in self.grid.items():
            best_val, best_gain = base, -1
            # scan 10 multipliers around base
            for mul in np.linspace(0.5, 1.8, 10):
                th = mul * base
                supp_p = self._support(name, th, pos)
                supp_n = self._support(name, th, neg)
                gain = supp_p - supp_n
                if gain > best_gain:
                    best_gain, best_val = gain, th
            best[name] = best_val
        self.learned = best
        return best

    # ---------- helpers ----------
    def _support(self, pred, th, group):
        cnt = 0
        fn = self._edge_fn(pred, th)
        for a, b in combinations(group, 2):
            cnt += fn(a, b)
        total = len(group) * (len(group) - 1) / 2
        return cnt / (total or 1)

    def _edge_fn(self, pred, th):
        def safe_get(node, key):
            val = node.get(key)
            return val if val is not None else None
        if pred == "larger_than":
            return lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "area") is not None and safe_get(b, "area") is not None
                and safe_get(a, "area") > th * safe_get(b, "area")
            )
        if pred == "aspect_sim":
            return lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "aspect_ratio") is not None and safe_get(b, "aspect_ratio") is not None
                and abs(safe_get(a, "aspect_ratio") - safe_get(b, "aspect_ratio")) < th
            )
        if pred == "near":
            return lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('centroid_valid', False)
                and b.get('feature_valid', {}).get('centroid_valid', False)
                and safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < th
            )
        if pred == "para":
            return lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('orientation_valid', False)
                and b.get('feature_valid', {}).get('orientation_valid', False)
                and safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < th
            )
        return lambda a, b: False

    def build_edge_fn(self):
        """Return a dict of edge lambdas using learned thresholds, with robust feature validity guards."""
        assert hasattr(self, "learned"), "Call fit() first."
        learned = self.learned
        def safe_get(node, key):
            val = node.get(key)
            return val if val is not None else None
        return {
            "larger_than": lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "area") is not None and safe_get(b, "area") is not None
                and safe_get(a, "area") > learned["larger_than"] * safe_get(b, "area")
            ),
            "aspect_sim": lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "aspect_ratio") is not None and safe_get(b, "aspect_ratio") is not None
                and abs(safe_get(a, "aspect_ratio") - safe_get(b, "aspect_ratio")) < learned["aspect_sim"]
            ),
            "near": lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('centroid_valid', False)
                and b.get('feature_valid', {}).get('centroid_valid', False)
                and safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < learned["near"]
            ),
            "para": lambda a, b: (
                a.get('geometry_valid', False) and b.get('geometry_valid', False)
                and a.get('feature_valid', {}).get('orientation_valid', False)
                and b.get('feature_valid', {}).get('orientation_valid', False)
                and safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < learned["para"]
            ),
        }
