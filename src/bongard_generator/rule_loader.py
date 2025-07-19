"""Rule loader for Bongard problems"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BongardRule:
    """Represents a Bongard problem rule."""
    description: str
    positive_features: Dict[str, any]
    negative_features: Dict[str, any] = None
    
    def __post_init__(self):
        if self.negative_features is None:
            self.negative_features = {}



# Robust import and normalization of ALL_BONGARD_RULES
try:
    from src.bongard_rules import ALL_BONGARD_RULES, BongardRule
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
    assert "SHAPE(TRIANGLE)" in RULE_LOOKUP, "Default rule SHAPE(TRIANGLE) missing from RULE_LOOKUP"
except Exception as e:
    logger.warning(f"Could not import from src.bongard_rules, using fallback rules: {e}")
    from dataclasses import dataclass
    @dataclass
    class BongardRule:
        def __init__(self, description: str, positive_features: Dict[str, any], negative_features: Dict[str, any] = None):
            self.description = description
            self.positive_features = positive_features
            self.negative_features = negative_features if negative_features is not None else {}
            # Canonical key mapping
            self.name = self._make_canonical_key(description, positive_features)

        def _make_canonical_key(self, description, features):
            # Map description/features to canonical dataset key
            desc = description.strip().upper()
            if desc.startswith("SHAPE("):
                return f"shape_{features['shape']}"
            elif desc.startswith("COUNT("):
                return f"count_eq_{features['count']}"
            elif desc.startswith("FILL("):
                return f"fill_{features['fill']}"
            elif desc.startswith("SPATIAL("):
                return f"{features['relation']}"
            elif desc.startswith("TOPO("):
                return f"{features['relation']}"
            else:
                # fallback: use lowercased description
                return desc.lower().replace('(', '_').replace(')', '').replace(' ', '_')
    ALL_RULES = [
        BongardRule("SHAPE(TRIANGLE)", {"shape": "triangle"}),
        BongardRule("SHAPE(SQUARE)", {"shape": "square"}),
        BongardRule("SHAPE(CIRCLE)", {"shape": "circle"}),
        BongardRule("SHAPE(PENTAGON)", {"shape": "pentagon"}),
        BongardRule("SHAPE(STAR)", {"shape": "star"}),
        BongardRule("COUNT(1)", {"count": 1}),
        BongardRule("COUNT(2)", {"count": 2}),
        BongardRule("COUNT(3)", {"count": 3}),
        BongardRule("COUNT(4)", {"count": 4}),
        BongardRule("FILL(SOLID)", {"fill": "solid"}),
        BongardRule("FILL(OUTLINE)", {"fill": "outline"}),
        BongardRule("FILL(STRIPED)", {"fill": "striped"}),
        BongardRule("SPATIAL(LEFT_OF)", {"relation": "left_of"}),
        BongardRule("SPATIAL(ABOVE)", {"relation": "above"}),
        BongardRule("TOPO(OVERLAP)", {"relation": "overlap"}),
        BongardRule("TOPO(NESTED)", {"relation": "nested"}),
    ]
    RULE_LOOKUP: Dict[str, BongardRule] = {r.description.strip().upper(): r for r in ALL_RULES}

def get_all_rules() -> List[BongardRule]:
    return ALL_RULES

def get_rule_lookup() -> Dict[str, BongardRule]:
    return RULE_LOOKUP.copy()

def get_rule_by_description(description: str) -> BongardRule:
    return RULE_LOOKUP[description.strip().upper()]

def get_all_rules() -> List[BongardRule]:
    """Get all available Bongard rules."""
    return ALL_RULES.copy()

def get_rule_lookup() -> Dict[str, BongardRule]:
    """Get the rule lookup dictionary."""
    return RULE_LOOKUP.copy()

def get_rule_by_description(description: str) -> Optional[BongardRule]:
    """Get a rule by its description."""
    return RULE_LOOKUP.get(description.strip().upper())

def validate_rules() -> bool:
    """Validate that all rules are properly formatted."""
    for rule in ALL_RULES:
        if not rule.description:
            logger.error(f"Rule missing description: {rule}")
            return False
        if not rule.positive_features:
            logger.error(f"Rule missing positive features: {rule.description}")
            return False
    return True

# Validate rules on import
if not validate_rules():
    logger.error("Rule validation failed!")
else:
    logger.info(f"Successfully loaded and validated {len(ALL_RULES)} rules")
