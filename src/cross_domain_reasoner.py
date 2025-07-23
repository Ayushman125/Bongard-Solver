"""
Cross-domain reasoner that fuses physics proxies with commonsense predicates
Phase 1 Module
"""

from enum import Enum
from dataclasses import dataclass
from src.commonsense_kb import CommonsenseKB
from typing import Dict

class ReasoningMode(Enum):
    PHYSICS_ONLY = "physics_only"
    COMMONSENSE_ONLY = "commonsense_only"
    FUSION = "fusion"

@dataclass
class ReasoningResult:
    conclusion: str
    confidence: float
    evidence: Dict
    reasoning_mode: ReasoningMode

class CrossDomainReasoner:
    """Fuses physics-based inferences with commonsense knowledge"""
    def __init__(self, kb_path: str = 'data/conceptnet_lite.json'):
        self.kb = CommonsenseKB(kb_path)
        self.fusion_weights = {'physics': 0.6, 'commonsense': 0.4}
    def reason_about_objects(self, physics_data: Dict, visual_features: Dict, query_context: str = "") -> ReasoningResult:
        reasoning_mode = self._select_reasoning_mode(physics_data, visual_features, query_context)
        if reasoning_mode == ReasoningMode.PHYSICS_ONLY:
            return self._physics_reasoning(physics_data, visual_features)
        elif reasoning_mode == ReasoningMode.COMMONSENSE_ONLY:
            return self._commonsense_reasoning(visual_features, query_context)
        else:
            return self._fusion_reasoning(physics_data, visual_features, query_context)
    def _select_reasoning_mode(self, physics: Dict, visuals: Dict, context: str):
        score = physics.get('stability_score', 0)
        ctx = context.lower()
        if 'why' in ctx or (score > 0.85):
            return ReasoningMode.PHYSICS_ONLY
        return ReasoningMode.COMMONSENSE_ONLY
    def _physics_reasoning(self, physics_data, visual_features):
        return ReasoningResult(
            conclusion="Physics-only conclusion",
            confidence=0.7,
            evidence=physics_data,
            reasoning_mode=ReasoningMode.PHYSICS_ONLY
        )
    def _commonsense_reasoning(self, visual_features, query_context):
        return ReasoningResult(
            conclusion="Commonsense-only conclusion",
            confidence=0.6,
            evidence=visual_features,
            reasoning_mode=ReasoningMode.COMMONSENSE_ONLY
        )
    def _fusion_reasoning(self, physics_data, visual_features, query_context):
        confidence = self.fusion_weights['physics'] * 0.7 + self.fusion_weights['commonsense'] * 0.6
        evidence = {'physics': physics_data, 'visual': visual_features, 'context': query_context}
        return ReasoningResult(
            conclusion="Fusion conclusion",
            confidence=confidence,
            evidence=evidence,
            reasoning_mode=ReasoningMode.FUSION
        )
