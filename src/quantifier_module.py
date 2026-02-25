"""
Quantifier module for ∀/∃ detection on repeated relations
Phase 1 Module - Enhanced Implementation
"""

from typing import List, Dict
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

class QuantifierType(Enum):
    UNIVERSAL = "∀"
    EXISTENTIAL = "∃"
    NONE = "none"

@dataclass
class QuantifierPattern:
    quantifier_type: QuantifierType
    variable: str
    predicate: str
    confidence: float
    supporting_evidence: List[Dict]

class QuantifierModule:
    """Detects universal (∀) and existential (∃) quantifiers in visual relations"""
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.min_instances_for_universal = 3
    def detect_quantifiers(self, relationships: List[Dict], objects: List[Dict]) -> List[QuantifierPattern]:
        patterns = []
        relation_groups = defaultdict(list)
        for rel in relationships:
            predicate = rel.get('predicate', 'unknown')
            relation_groups[predicate].append(rel)
        for predicate, relations in relation_groups.items():
            coverage = len(relations) / max(1, len(objects))
            confidence = min(1.0, coverage)
            # Universal quantifier: predicate holds for all (above threshold)
            if len(relations) >= self.min_instances_for_universal and coverage > 0.8:
                patterns.append(QuantifierPattern(
                    quantifier_type=QuantifierType.UNIVERSAL,
                    variable="object",
                    predicate=predicate,
                    confidence=confidence,
                    supporting_evidence=relations
                ))
            # Existential quantifier: predicate holds for at least one
            if len(relations) > 0:
                patterns.append(QuantifierPattern(
                    quantifier_type=QuantifierType.EXISTENTIAL,
                    variable="object",
                    predicate=predicate,
                    confidence=confidence,
                    supporting_evidence=relations
                ))
        return [p for p in patterns if p.confidence >= self.confidence_threshold]
