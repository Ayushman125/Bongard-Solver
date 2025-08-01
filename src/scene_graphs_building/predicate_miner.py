import numpy as np
from itertools import combinations
from sklearn.metrics import mutual_info_score

DEFAULT_CANDIDATES = {
    "larger_than": 1.15,
    "aspect_sim": 0.20,
    "near": 32.0,
    "para": 12.0,
    "length_sim": 0.20,
    "orientation_sim": 12.0,
    "centroid_dist": 32.0,
}

import logging
import difflib

class PredicateMiner:
    def __init__(self, search_grid=None, adaptive_thresholds=None):
        self.grid = search_grid or DEFAULT_CANDIDATES.copy()
        self.adaptive_thresholds = adaptive_thresholds  # Optional: pass AdaptivePredicateThresholds instance

    def fit(self, objects):
        """
        Pick thresholds that maximise Δ = support_pos – support_neg. Handles all primitives and new features.
        Integrates adaptive thresholds/statistics if available.
        Logs skipped/invalid features for diagnostics.
        """
        pos = [o for o in objects if o.get("shape_label") == "positive"]
        neg = [o for o in objects if o.get("shape_label") == "negative"]
        best = {}
        skipped = []
        # Expand candidate features for SOTA: include stroke, curvature, programmatic, KB, global stats
        all_candidates = set(self.grid.keys()) | {
            "curvature_sim", "stroke_count_sim", "programmatic_sim", "kb_sim", "global_stat_sim"
        }
        # Optionally add multi-feature predicates
        all_candidates |= {"near_and_para", "same_program_and_near"}
        for name in all_candidates:
            base = self.grid.get(name, 1.0)
            # Use adaptive threshold if available
            if self.adaptive_thresholds and name in self.adaptive_thresholds:
                base = self.adaptive_thresholds[name]
            best_val, best_gain = base, -1
            for mul in np.linspace(0.5, 1.8, 10):
                th = mul * base
                supp_p = self._support(name, th, pos)
                supp_n = self._support(name, th, neg)
                gain = supp_p - supp_n
                if gain > best_gain:
                    best_gain, best_val = gain, th
            if best_gain < 0.01:
                skipped.append(name)
            best[name] = best_val
        self.learned = best
        if skipped:
            logging.warning(f"PredicateMiner: Skipped low-gain predicates: {skipped}")
        if len(skipped) == len(all_candidates):
            logging.error("PredicateMiner: All predicates skipped/invalid! Check data pipeline.")
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
        # Area-based predicates (polygons only, but permissive for lines/arcs)
        if pred == "larger_than":
            return lambda a, b: (
                a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon'
                and a.get('feature_valid', {}).get('area_valid', a.get('object_type') != 'polygon')
                and b.get('feature_valid', {}).get('area_valid', b.get('object_type') != 'polygon')
                and safe_get(a, "area") is not None and safe_get(b, "area") is not None
                and safe_get(a, "area") > th * safe_get(b, "area")
            )
        if pred == "aspect_sim":
            return lambda a, b: (
                a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon'
                and a.get('feature_valid', {}).get('area_valid', a.get('object_type') != 'polygon')
                and b.get('feature_valid', {}).get('area_valid', b.get('object_type') != 'polygon')
                and safe_get(a, "aspect_ratio") is not None and safe_get(b, "aspect_ratio") is not None
                and abs(safe_get(a, "aspect_ratio") - safe_get(b, "aspect_ratio")) < th
            )
        # For all open/degenerate shapes: always compute and use length, centroid, orientation if present
        if pred == "near":
            return lambda a, b: (
                safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < th
            )
        if pred == "para":
            return lambda a, b: (
                safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < th
            )
        if pred == "length_sim":
            return lambda a, b: (
                safe_get(a, "length") is not None and safe_get(b, "length") is not None
                and abs(safe_get(a, "length") - safe_get(b, "length")) < th
            )
        if pred == "orientation_sim":
            return lambda a, b: (
                safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < th
            )
        if pred == "centroid_dist":
            return lambda a, b: (
                safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < th
            )
        # SOTA: Add new feature predicates
        if pred == "curvature_sim":
            return lambda a, b: (
                safe_get(a, "curvature") is not None and safe_get(b, "curvature") is not None
                and abs(safe_get(a, "curvature") - safe_get(b, "curvature")) < th
            )
        if pred == "stroke_count_sim":
            return lambda a, b: (
                safe_get(a, "stroke_count") is not None and safe_get(b, "stroke_count") is not None
                and abs(safe_get(a, "stroke_count") - safe_get(b, "stroke_count")) < th
            )
        # Soft programmatic/KB matching (Levenshtein/difflib)
        if pred == "programmatic_sim":
            return lambda a, b: (
                safe_get(a, "programmatic_label") is not None and safe_get(b, "programmatic_label") is not None
                and (a["programmatic_label"] == b["programmatic_label"] or
                     difflib.SequenceMatcher(None, str(a["programmatic_label"]), str(b["programmatic_label"])).ratio() > 0.85)
            )
        if pred == "kb_sim":
            return lambda a, b: (
                safe_get(a, "kb_concept") is not None and safe_get(b, "kb_concept") is not None
                and (a["kb_concept"] == b["kb_concept"] or
                     difflib.SequenceMatcher(None, str(a["kb_concept"]), str(b["kb_concept"])).ratio() > 0.85)
            )
        if pred == "global_stat_sim":
            return lambda a, b: (
                safe_get(a, "global_stat") is not None and safe_get(b, "global_stat") is not None
                and abs(safe_get(a, "global_stat") - safe_get(b, "global_stat")) < th
            )
        # Multi-feature predicates
        if pred == "near_and_para":
            return lambda a, b: (
                safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < th
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < 15.0
            )
        if pred == "same_program_and_near":
            return lambda a, b: (
                safe_get(a, "programmatic_label") is not None and safe_get(b, "programmatic_label") is not None
                and a["programmatic_label"] == b["programmatic_label"]
                and safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < th
            )
        return lambda a, b: False

    def build_edge_fn(self, type_aware=False):
        """Return a dict of edge lambdas using learned thresholds, with robust feature validity guards for all primitives.
        If type_aware=True, enable type-aware predicate logic (future extension).
        """
        assert hasattr(self, "learned"), "Call fit() first."
        learned = self.learned
        def safe_get(node, key):
            val = node.get(key)
            return val if val is not None else None
        # If type_aware is True, you can add type-specific logic here in the future
        # For now, fallback to the same logic as before
        return {
            # Area-based predicates (polygons only)
            "larger_than": lambda a, b: (
                a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon'
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "area") is not None and safe_get(b, "area") is not None
                and safe_get(a, "area") > learned["larger_than"] * safe_get(b, "area")
            ),
            "aspect_sim": lambda a, b: (
                a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon'
                and a.get('feature_valid', {}).get('area_valid', False)
                and b.get('feature_valid', {}).get('area_valid', False)
                and safe_get(a, "aspect_ratio") is not None and safe_get(b, "aspect_ratio") is not None
                and abs(safe_get(a, "aspect_ratio") - safe_get(b, "aspect_ratio")) < learned["aspect_sim"]
            ),
            # For all open/degenerate shapes: always compute and use length, centroid, orientation if present
            "near": lambda a, b: (
                safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < learned["near"]
            ),
            "para": lambda a, b: (
                safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < learned["para"]
            ),
            "length_sim": lambda a, b: (
                safe_get(a, "length") is not None and safe_get(b, "length") is not None
                and abs(safe_get(a, "length") - safe_get(b, "length")) < learned["length_sim"]
            ),
            "orientation_sim": lambda a, b: (
                safe_get(a, "orientation") is not None and safe_get(b, "orientation") is not None
                and abs(safe_get(a, "orientation") - safe_get(b, "orientation")) < learned["orientation_sim"]
            ),
            "centroid_dist": lambda a, b: (
                safe_get(a, "centroid") is not None and safe_get(b, "centroid") is not None
                and all(isinstance(x, (int, float)) for x in safe_get(a, "centroid") + safe_get(b, "centroid"))
                and np.linalg.norm(np.array(safe_get(a, "centroid")) - np.array(safe_get(b, "centroid"))) < learned["centroid_dist"]
            ),
            # SOTA: Add new feature predicates
            "curvature_sim": lambda a, b: (
                safe_get(a, "curvature") is not None and safe_get(b, "curvature") is not None
                and abs(safe_get(a, "curvature") - safe_get(b, "curvature")) < learned.get("curvature_sim", 1.0)
            ),
            "stroke_count_sim": lambda a, b: (
                safe_get(a, "stroke_count") is not None and safe_get(b, "stroke_count") is not None
                and abs(safe_get(a, "stroke_count") - safe_get(b, "stroke_count")) < learned.get("stroke_count_sim", 1.0)
            ),
            "programmatic_sim": lambda a, b: (
                safe_get(a, "programmatic_label") is not None and safe_get(b, "programmatic_label") is not None
                and a["programmatic_label"] == b["programmatic_label"]
            ),
            "kb_sim": lambda a, b: (
                safe_get(a, "kb_concept") is not None and safe_get(b, "kb_concept") is not None
                and a["kb_concept"] == b["kb_concept"]
            ),
            "global_stat_sim": lambda a, b: (
                safe_get(a, "global_stat") is not None and safe_get(b, "global_stat") is not None
                and abs(safe_get(a, "global_stat") - safe_get(b, "global_stat")) < learned.get("global_stat_sim", 1.0)
            ),
        }
