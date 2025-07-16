# Folder: bongard_solver/src/
# File: causal.py

import logging
from typing import List, Dict, Any, Tuple, Optional
import random

# Import BongardRule from bongard_rules
try:
    from src.bongard_rules import BongardRule
except ImportError:
    logging.warning("Could not import BongardRule in Causal. Using dummy BongardRule.")
    class BongardRule:
        def __init__(self, name, description, program_ast, logical_facts, is_positive_rule=True):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
            self.is_positive_rule = is_positive_rule
        def __repr__(self): return self.name

logger = logging.getLogger(__name__)

class CausalFilter:
    """
    Causal Filtering module for Bongard problems.
    This module takes a list of candidate rules (e.g., from ILP) and
    applies heuristic causal filtering to select or prioritize rules
    that are more likely to represent the underlying causal mechanism
    of the Bongard problem.
    """
    def __init__(self):
        logger.info("CausalFilter initialized.")

    @classmethod
    def filter(cls, candidate_rules: List[BongardRule], causal_strengths: Optional[Dict[str, float]] = None) -> List[BongardRule]:
        """
        Filters and prioritizes candidate rules based on heuristic causal principles.
        Integrates causal strengths as priors.
        Args:
            candidate_rules (List[BongardRule]): A list of BongardRule objects proposed by ILP.
            causal_strengths (Optional[Dict[str, float]]): A dictionary mapping rule component
                                                            names (e.g., 'SHAPE', 'LEFT_OF', 'circle')
                                                            to their learned causal strengths (0.0 to 1.0).
                                                            These could come from a separate causal learning module.
        Returns:
            List[BongardRule]: A filtered and potentially re-ordered list of rules.
        """
        logger.info(f"CausalFilter: Filtering {len(candidate_rules)} candidate rules.")
        
        if not candidate_rules:
            logger.warning("CausalFilter: No candidate rules to filter.")
            return []

        # If no causal strengths are provided, use uniform strengths (or a default)
        if causal_strengths is None:
            causal_strengths = collections.defaultdict(lambda: 0.5) # Default neutral strength
            logger.info("CausalFilter: No explicit causal strengths provided. Using default neutral strengths.")

        filtered_rules: List[BongardRule] = []

        def get_rule_complexity(rule: BongardRule) -> int:
            """Calculates a simple complexity metric for a rule."""
            num_facts = len(rule.logical_facts) if rule.logical_facts else 0
            
            def count_nodes(node_dict: Dict[str, Any]) -> int:
                count = 1
                if "args" in node_dict:
                    for arg in node_dict["args"]:
                        count += count_nodes(arg)
                return count
            
            ast_node_count = 0
            if rule.program_ast and isinstance(rule.program_ast, list) and rule.program_ast:
                ast_node_count = count_nodes(rule.program_ast[0])
            
            return num_facts + ast_node_count

        def get_causal_score(rule: BongardRule, strengths: Dict[str, float]) -> float:
            """
            Calculates a causal score for a rule based on the strengths of its components.
            Higher score means more causally plausible.
            """
            score = 0.0
            num_components = 0
            
            # Parse rule description or AST to extract components (e.g., 'SHAPE', 'circle', 'LEFT_OF')
            # This is a simplified parsing. A full parser would be needed for robust extraction.
            components_in_rule = set()
            
            # Attempt to extract from logical facts
            for fact_str in rule.logical_facts:
                parts = fact_str.replace('(', ',').replace(')', '').split(',')
                op = parts[0].strip().upper()
                components_in_rule.add(op)
                if len(parts) > 1:
                    for arg in parts[1:]:
                        clean_arg = arg.strip().lower()
                        if clean_arg and not clean_arg.startswith('obj_') and clean_arg != 'o': # Exclude object variables
                            components_in_rule.add(clean_arg)
            
            # Sum up strengths of identified components
            for component_name in components_in_rule:
                score += strengths.get(component_name, 0.5) # Default to 0.5 if not in strengths
                num_components += 1
            
            return score / num_components if num_components > 0 else 0.0

        # Assign a combined score: (causal_score * causal_weight) - (complexity * complexity_weight)
        # Or, simply use causal score for re-ranking after initial complexity filtering
        
        # Step 1: Filter by a minimum causal score (if applicable)
        # This can be used to discard rules that are deemed very implausible.
        # For now, we'll just use it for ranking.

        # Step 2: Calculate a combined score and re-rank
        scored_rules = []
        for rule in candidate_rules:
            complexity = get_rule_complexity(rule)
            causal_score = get_causal_score(rule, causal_strengths)
            
            # Simple combined score: higher causal_score is better, lower complexity is better
            # You might want to tune weights for these components.
            # For now, let's prioritize causal score, then complexity.
            combined_score = causal_score - (complexity * 0.01) # Small penalty for complexity
            
            scored_rules.append((combined_score, rule))
        
        # Sort rules by combined score in descending order
        sorted_rules = sorted(scored_rules, key=lambda x: x[0], reverse=True)
        
        # Extract just the rules
        filtered_rules = [rule for score, rule in sorted_rules]

        # Heuristic 3: Deduplicate identical rules (based on logical facts)
        # This should ideally happen earlier, but can be a final pass.
        unique_rules_by_facts: Dict[Tuple[str, ...], BongardRule] = {}
        for rule in filtered_rules: # Iterate through already sorted rules
            facts_tuple = tuple(sorted(rule.logical_facts))
            if facts_tuple not in unique_rules_by_facts:
                unique_rules_by_facts[facts_tuple] = rule
            else:
                logger.debug(f"CausalFilter: Deduplicated rule: {rule.name} (identical to {unique_rules_by_facts[facts_tuple].name})")
        
        filtered_rules = list(unique_rules_by_facts.values())

        logger.info(f"CausalFilter: Filtered down to {len(filtered_rules)} rules.")
        return filtered_rules

