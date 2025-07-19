"""Rule loader for Bongard problems"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

@dataclass
class BongardRule:
    """Represents a Bongard problem rule."""
    description: str
    positive_features: Dict[str, any] = field(default_factory=dict)
    negative_features: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        if self.negative_features is None:
            self.negative_features = {}
        if self.positive_features is None:
            self.positive_features = {}
        # Fallback logic: set .name if not present
        if not hasattr(self, 'name'):
            self.name = self._make_canonical_key(self.description, self.positive_features)

    def _make_canonical_key(self, description, features):
        desc = description.strip().upper()
        if desc.startswith("SHAPE("):
            return f"shape_{features.get('shape', '')}"
        elif desc.startswith("COUNT("):
            return f"count_eq_{features.get('count', '')}"
        elif desc.startswith("FILL("):
            return f"fill_{features.get('fill', '')}"
        elif desc.startswith("SPATIAL("):
            return f"{features.get('relation', '')}"
        elif desc.startswith("TOPO("):
            return f"{features.get('relation', '')}"
        else:
            return desc.lower().replace('(', '_').replace(')', '').replace(' ', '_')



# Robust import and normalization of ALL_BONGARD_RULES
try:
    from src.bongard_rules import ALL_BONGARD_RULES
    # Normalize ALL_BONGARD_RULES to a list if it's a dict
    if isinstance(ALL_BONGARD_RULES, dict):
        _rules = list(ALL_BONGARD_RULES.values())
    else:
        _rules = ALL_BONGARD_RULES
    assert _rules, "No Bongard rules found!"
    assert hasattr(_rules[0], "description"), "Rule objects must have .description"
    ALL_RULES = _rules
    RULE_LOOKUP: Dict[str, BongardRule] = {r.description.strip().upper(): r for r in ALL_RULES}
    # Quick sanity assert for validation harness
    # assert "SHAPE(TRIANGLE)" in RULE_LOOKUP, "Default rule SHAPE(TRIANGLE) missing from RULE_LOOKUP"
except Exception as e:
    logger.warning(f"Could not import from src.bongard_rules, using fallback rules: {e}")
    # Use the unified BongardRule dataclass already defined above

    _fallback_rules = [
        {"description": "SHAPE(TRIANGLE)", "positive_features": {"shape": "triangle"}},
        {"description": "SHAPE(SQUARE)", "positive_features": {"shape": "square"}},
        {"description": "SHAPE(CIRCLE)", "positive_features": {"shape": "circle"}},
        {"description": "SHAPE(PENTAGON)", "positive_features": {"shape": "pentagon"}},
        {"description": "SHAPE(STAR)", "positive_features": {"shape": "star"}},
        {"description": "COUNT(1)", "positive_features": {"count": 1}},
        {"description": "COUNT(2)", "positive_features": {"count": 2}},
        {"description": "COUNT(3)", "positive_features": {"count": 3}},
        {"description": "COUNT(4)", "positive_features": {"count": 4}},
        {"description": "FILL(SOLID)", "positive_features": {"fill": "solid"}},
        {"description": "FILL(OUTLINE)", "positive_features": {"fill": "outline"}},
        {"description": "FILL(STRIPED)", "positive_features": {"fill": "striped"}},
        {"description": "SPATIAL(LEFT_OF)", "positive_features": {"relation": "left_of"}},
        {"description": "SPATIAL(ABOVE)", "positive_features": {"relation": "above"}},
        {"description": "TOPO(OVERLAP)", "positive_features": {"relation": "overlap"}},
        {"description": "TOPO(NESTED)", "positive_features": {"relation": "nested"}},
    ]
    ALL_RULES = []
    for rule_kwargs in _fallback_rules:
        rule = BongardRule(**rule_kwargs)
        rule.name = rule._make_canonical_key(rule.description, rule.positive_features)
        ALL_RULES.append(rule)
    RULE_LOOKUP: Dict[str, BongardRule] = {r.description.strip().upper(): r for r in ALL_RULES}

def get_all_rules() -> List[BongardRule]:
    return ALL_RULES.copy()

def get_rule_lookup() -> Dict[str, BongardRule]:
    return RULE_LOOKUP.copy()

def get_rule_by_description(description: str) -> Optional[BongardRule]:
    return RULE_LOOKUP.get(description.strip().upper())

def validate_rules() -> bool:
    """Validate that all rules are properly formatted."""
    for rule in ALL_RULES:
        # Defensive: ensure description exists
        if not hasattr(rule, 'description') or not rule.description:
            return False
        # Defensive: ensure positive_features exists and is a dict
        if not hasattr(rule, 'positive_features') or not isinstance(rule.positive_features, dict):
            return False
        # Defensive: ensure negative_features exists and is a dict
        if not hasattr(rule, 'negative_features') or not isinstance(rule.negative_features, dict):
            return False
        # Defensive: ensure .name exists
        if not hasattr(rule, 'name') or not rule.name:
            rule.name = rule._make_canonical_key(rule.description, rule.positive_features)
    return True

# Validate rules on import
if not validate_rules():
    pass
