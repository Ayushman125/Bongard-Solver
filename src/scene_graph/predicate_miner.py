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
        if pred == "larger_than":
            return lambda a, b: a["area"] > th * b["area"]
        if pred == "aspect_sim":
            return lambda a, b: abs(a["aspect_ratio"] - b["aspect_ratio"]) < th
        if pred == "near":
            return lambda a, b: np.linalg.norm(np.array(a["centroid"]) - np.array(b["centroid"])) < th
        if pred == "para":
            return lambda a, b: abs(a["orientation"] - b["orientation"]) < th
        return lambda a, b: False

    def build_edge_fn(self):
        """Return a dict of edge lambdas using learned thresholds."""
        assert hasattr(self, "learned"), "Call fit() first."
        learned = self.learned
        return {
            "larger_than": lambda a, b: a["area"] > learned["larger_than"] * b["area"],
            "aspect_sim": lambda a, b: abs(a["aspect_ratio"] - b["aspect_ratio"]) < learned["aspect_sim"],
            "near": lambda a, b: np.linalg.norm(np.array(a["centroid"]) - np.array(b["centroid"])) < learned["near"],
            "para": lambda a, b: abs(a["orientation"] - b["orientation"]) < learned["para"],
        }
