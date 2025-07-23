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

class PhysicsInference:
    """Batched physics-proxy calculations with GPU acceleration where possible"""
    def __init__(self, kb_path: str = 'data/conceptnet_lite.json'):
        self.kb = CommonsenseKB(kb_path)
        self.profiler = TaskProfiler()
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
    def compute_stability_batch(self, masks: List[np.ndarray], coms: np.ndarray = None) -> List[Dict]:
        if coms is None:
            coms = self.extract_center_of_mass_batch(masks)
        stabilities = []
        for mask, com in zip(masks, coms):
            stability = self._compute_single_stability(mask, com)
            stabilities.append(stability)
        return stabilities
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
    def compute_affordances(self, masks: List[np.ndarray], predicates: List[str]) -> List[Dict]:
        affordances = []
        for mask, predicate in zip(masks, predicates):
            context = [str(np.sum(mask))]
            kb_result = self.kb.query(predicate, context)
            affordances.append(kb_result)
        return affordances
