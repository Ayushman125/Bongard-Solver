# Folder: bongard_solver/
# File: rule_inducer.py

import logging
import random
from typing import Dict, Any, List, Tuple, Optional

# Import Slipnet
from slipnet import Slipnet

# Import BongardRule (assuming it's in bongard_rules.py)
from bongard_rules import BongardRule, ALL_BONGARD_RULES

logger = logging.getLogger(__name__)

class RuleInducer:
    """
    Responsible for proposing, refining, and testing Bongard rule fragments
    based on the current state of the workspace and Slipnet activations.
    In a full Copycat, these actions would be triggered by Codelets.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Slipnet):
        """
        Initializes the RuleInducer.
        Args:
            config (Dict[str, Any]): Global configuration.
            slipnet (Slipnet): The Slipnet instance for concept activation.
        """
        self.config = config
        self.slipnet = slipnet
        self.known_bongard_rules = list(ALL_BONGARD_RULES.values()) # Access to predefined rules
        logger.info("RuleInducer initialized.")

    def propose_rule_fragments(self, workspace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Proposes new rule fragments based on active concepts in the Slipnet
        and patterns observed in the workspace scene graphs.
        Args:
            workspace (Dict[str, Any]): The current state of the symbolic workspace.
        Returns:
            List[Dict[str, Any]]: A list of newly proposed rule fragments.
        """
        proposed_fragments = []
        active_concepts = self.slipnet.get_active_nodes(
            threshold=self.config['slipnet_config']['activation_threshold']
        )
        
        logger.debug(f"RuleInducer: Proposing rule fragments based on active concepts: {active_concepts.keys()}")

        # Example: Propose rules based on highly active attribute concepts
        for concept_name, activation in active_concepts.items():
            if activation > 0.5: # Only consider highly active concepts
                # Simple rule: "All objects have X attribute"
                if concept_name in ['circle', 'square', 'red', 'blue', 'small', 'large']:
                    rule_text = f"All objects are {concept_name}"
                    # A rule fragment could be a dictionary representing its structure
                    proposed_fragments.append({
                        'type': 'attribute_constraint',
                        'attribute': concept_name,
                        'rule_text': rule_text,
                        'confidence': activation # Initial confidence based on activation
                    })
                    logger.debug(f"  Proposed rule fragment: '{rule_text}'")

                # Simple rule: "There exists an X object"
                if concept_name in ['circle', 'square', 'red', 'blue']:
                    rule_text = f"Exists a {concept_name} object"
                    proposed_fragments.append({
                        'type': 'existence_constraint',
                        'attribute': concept_name,
                        'rule_text': rule_text,
                        'confidence': activation * 0.8 # Slightly lower confidence
                    })
                    logger.debug(f"  Proposed rule fragment: '{rule_text}'")

                # Example: Propose rules based on highly active relation concepts
                if concept_name in ['left_of', 'above', 'same_shape', 'same_color']:
                    rule_text = f"Objects are {concept_name}"
                    proposed_fragments.append({
                        'type': 'relational_constraint',
                        'relation': concept_name,
                        'rule_text': rule_text,
                        'confidence': activation * 0.9
                    })
                    logger.debug(f"  Proposed rule fragment: '{rule_text}'")
        
        # Limit the number of proposed fragments to avoid explosion
        return random.sample(proposed_fragments, min(len(proposed_fragments), 5)) # Propose max 5

    def test_rule_fragments(self, workspace: Dict[str, Any]):
        """
        Tests existing rule fragments against the current Bongard problem
        (query and support sets) and updates their confidence.
        Successful tests can reinforce concepts in the Slipnet.
        Args:
            workspace (Dict[str, Any]): The current state of the symbolic workspace.
        """
        query_sg1 = workspace['query_scene_graph_view1']
        query_sg2 = workspace['query_scene_graph_view2']
        support_sgs = workspace['support_scene_graphs']
        current_rule_fragments = workspace['current_rule_fragments']

        if not current_rule_fragments:
            logger.debug("RuleInducer: No rule fragments to test.")
            return

        logger.debug(f"RuleInducer: Testing {len(current_rule_fragments)} rule fragments.")

        for fragment in current_rule_fragments:
            # Simulate testing the fragment against the problem
            # This is a placeholder for actual rule evaluation logic.
            # In a real system, this would involve a symbolic rule interpreter.

            # For now, let's just check if the rule fragment's text matches
            # any part of a known Bongard rule.
            
            # This is a very simplified "test"
            is_consistent_with_problem = False
            if 'rule_text' in fragment:
                for known_rule in self.known_bongard_rules:
                    if fragment['rule_text'].lower() in known_rule.description.lower():
                        is_consistent_with_problem = True
                        break
            
            # Update confidence based on consistency
            if is_consistent_with_problem:
                fragment['confidence'] = min(1.0, fragment.get('confidence', 0.0) + 0.1) # Reinforce
                logger.debug(f"  Fragment '{fragment.get('rule_text', 'N/A')}' reinforced. New confidence: {fragment['confidence']:.2f}")
                
                # Reinforce concepts related to this successful fragment in Slipnet
                if 'attribute' in fragment and fragment['attribute'] in self.slipnet.nodes:
                    self.slipnet.activate_node(fragment['attribute'], activation_value=fragment['confidence'])
                if 'relation' in fragment and fragment['relation'] in self.slipnet.nodes:
                    self.slipnet.activate_node(fragment['relation'], activation_value=fragment['confidence'])
                self.slipnet.activate_node('rule', activation_value=fragment['confidence'] * 0.5) # Reinforce 'rule' concept
            else:
                fragment['confidence'] = max(0.0, fragment.get('confidence', 0.0) - 0.05) # Decay
                logger.debug(f"  Fragment '{fragment.get('rule_text', 'N/A')}' decayed. New confidence: {fragment['confidence']:.2f}")

