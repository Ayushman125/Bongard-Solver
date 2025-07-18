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


# Try to import from the real bongard_rules, fall back to defaults
try:
    from src.bongard_rules import ALL_BONGARD_RULES, BongardRule as RealBongardRule
    # Normalize ALL_BONGARD_RULES to a list if it's a dict
    if isinstance(ALL_BONGARD_RULES, dict):
        _rules_seq = list(ALL_BONGARD_RULES.values())
    else:
        _rules_seq = ALL_BONGARD_RULES
    # Sanity check
    if not _rules_seq:
        raise ImportError("No Bongard rules loaded!")
    first_rule = _rules_seq[0] if isinstance(_rules_seq, list) else next(iter(_rules_seq))
    if hasattr(first_rule, "description"):
        logger.info("Successfully loaded rules from src.bongard_rules")
        RULES = _rules_seq
    else:
        raise ImportError("Rules format mismatch: missing 'description' attribute")
except ImportError:
    logger.warning("Could not import from src.bongard_rules, using fallback rules")
    # Fallback rules
    RULES = [
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

# Build a lookup for fast access by description
RULE_LOOKUP: Dict[str, BongardRule] = {
    r.description.strip().upper(): r for r in RULES
}

# Ensure default rule exists
if 'SHAPE(TRIANGLE)' not in RULE_LOOKUP:
    logger.warning("Default rule SHAPE(TRIANGLE) missing, adding it")
    default_rule = BongardRule("SHAPE(TRIANGLE)", {"shape": "triangle"})
    RULES.append(default_rule)
    RULE_LOOKUP['SHAPE(TRIANGLE)'] = default_rule

def get_all_rules() -> List[BongardRule]:
    """Get all available Bongard rules."""
    return RULES.copy()

def get_rule_lookup() -> Dict[str, BongardRule]:
    """Get the rule lookup dictionary."""
    return RULE_LOOKUP.copy()

def get_rule_by_description(description: str) -> Optional[BongardRule]:
    """Get a rule by its description."""
    return RULE_LOOKUP.get(description.strip().upper())

def validate_rules() -> bool:
    """Validate that all rules are properly formatted."""
    for rule in RULES:
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
    logger.info(f"Successfully loaded and validated {len(RULES)} rules")
