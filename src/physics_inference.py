"""
Batched COM, stability, affordance calculations with commonsense KB lookups
Phase 1 Module
"""

import numpy as np
from scipy.ndimage import center_of_mass
import cv2
from src.commonsense_kb import CommonsenseKB
from integration.task_profiler import TaskProfiler
from typing import List, Dict

# Additional imports for geometry and file handling
from shapely.geometry import Polygon
import pymunk
import re
import os
import logging

class PhysicsInference:
    """Batched physics-proxy calculations with GPU acceleration where possible"""
    def __init__(self, kb_path: str = 'data/conceptnet_lite.json'):
        self.kb = CommonsenseKB(kb_path)
        self.profiler = TaskProfiler()

    # --- LOGO Parsing ---
    def parse_logo_script(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0
        vertices = [(x, y)]
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = pattern.match(line)
            if not match:
                continue
            cmd, val = match.group(1), float(match.group(2))
            if cmd == 'F':
                rad = np.deg2rad(angle)
                x += val * np.cos(rad)
                y += val * np.sin(rad)
            elif cmd == 'B':
                rad = np.deg2rad(angle)
                x -= val * np.cos(rad)
                y -= val * np.sin(rad)
            elif cmd == 'R':
                angle -= val
                angle %= 360
            elif cmd == 'L':
                angle += val
                angle %= 360
            vertices.append((x, y))
        return vertices
    def extract_center_of_mass_batch(self, masks: List[np.ndarray]) -> np.ndarray:
        coms = []
        for mask in masks:
            if mask.sum() == 0:
                h, w = mask.shape
                com = [h/2, w/2]
            else:
                com = list(center_of_mass(mask))
            coms.append(com)
        return np.array(coms)

    # --- Polygon Geometry Features ---
    def polygon_from_vertices(self, vertices):
        try:
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly
        except Exception as e:
            print(f"Polygon creation failed: {e}")
            return None

    def centroid(self, poly):
        return (poly.centroid.x, poly.centroid.y)

    def area(self, poly):
        return poly.area

    def is_convex(self, poly):
        return poly.convex_hull.equals(poly)

    def symmetry_score(self, vertices):
        poly = Polygon(vertices)
        cx, cy = poly.centroid.x, poly.centroid.y
        reflected = [(2*cx - x, y) for (x, y) in vertices]
        orig = np.array(vertices)
        refl = np.array(reflected)
        rmse = np.sqrt(np.mean((orig - refl) ** 2))
        return rmse

    def moment_of_inertia(self, vertices):
        verts = [(float(x), float(y)) for (x, y) in vertices]
        try:
            moment = pymunk.moment_for_poly(1.0, verts)
        except Exception:
            moment = 0.0
        return moment
    def compute_stability_batch(self, masks: List[np.ndarray], coms: np.ndarray = None) -> List[Dict]:
        if coms is None:
            coms = self.extract_center_of_mass_batch(masks)
        stabilities = []
        for mask, com in zip(masks, coms):
            stability = self._compute_single_stability(mask, com)
            stabilities.append(stability)
        return stabilities

    # --- Meta-label Extraction ---
    def extract_problem_type(self, problem_path):
        for pt in ['Freeform', 'Basic', 'Abstract']:
            if pt.lower() in problem_path.lower():
                return pt
        return 'Unknown'

    def enrich_problem_dict(self, problem_dict, problem_path):
        problem_type = self.extract_problem_type(problem_path)
        problem_dict['problem_type'] = problem_type
        return problem_dict
    def _compute_single_stability(self, mask: np.ndarray, com: np.ndarray) -> Dict:
        h, w = mask.shape
        support_base = []
        for x in range(w):
            col = mask[:, x]
            if col.sum() > 0:
                bottom_y = np.where(col > 0)[0][-1]
                support_base.append([bottom_y, x])
        if len(support_base) == 0:
            support_width = 0
            stability_score = 0.0
        else:
            support_width = max([x for _, x in support_base]) - min([x for _, x in support_base]) + 1
            stability_score = float(support_width) / w
        return {
            'com': com,
            'is_stable': stability_score > 0.3,
            'support_width': float(support_width),
            'stability_score': float(stability_score)
        }

    # --- Manual Review Flagging ---
    def validate_polygon(self, poly):
        if not poly.is_valid:
            logging.warning(f"Invalid polygon detected: {poly.wkt}")
            return False
        return True

    def log_for_review(self, problem_id, filename, reason, review_log='data/flagged_cases.txt'):
        with open(review_log, 'a') as f:
            f.write(f"{problem_id},{filename},{reason}\n")
    def compute_affordances(self, masks: List[np.ndarray], predicates: List[str]) -> List[Dict]:
        affordances = []
        for mask, predicate in zip(masks, predicates):
            context = [str(np.sum(mask))]
            kb_result = self.kb.query(predicate, context)
            affordances.append(kb_result)
        return affordances
