# Folder: bongard_solver/
# File: symbolic_engine.py

import logging
from typing import Dict, Any, List, Tuple, Optional
import json

# Import Slipnet
from slipnet import Slipnet

# Import BongardRule (assuming it's in bongard_rules.py)
from bongard_rules import BongardRule, ALL_BONGARD_RULES

logger = logging.getLogger(__name__)

class SymbolicEngine:
    """
    The core symbolic reasoning engine.
    It analyzes inferred scene graphs, activates Slipnet concepts,
    and interacts with the RuleInducer to form and test Bongard rules.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Slipnet, rule_inducer: Any):
        """
        Initializes the SymbolicEngine.
        Args:
            config (Dict[str, Any]): Global configuration.
            slipnet (Slipnet): The Slipnet instance for concept activation.
            rule_inducer (Any): The RuleInducer instance for rule formation.
        """
        self.config = config
        self.slipnet = slipnet
        self.rule_inducer = rule_inducer # RuleInducer will be passed here
        
        self.workspace: Dict[str, Any] = {
            'query_scene_graph_view1': None,
            'query_scene_graph_view2': None,
            'support_scene_graphs': [],
            'current_rule_fragments': [], # Evolving rule fragments
            'active_concepts': {}, # Snapshot of active Slipnet concepts
            'step': 0
        }
        logger.info("SymbolicEngine initialized.")

    def analyze_scene_graph(self, inferred_scene_graph: Dict[str, Any]):
        """
        Analyzes an inferred scene graph and activates relevant concepts in the Slipnet.
        Args:
            inferred_scene_graph (Dict[str, Any]): The scene graph inferred by the PerceptionModule.
        """
        if not self.slipnet:
            logger.warning("Slipnet not available for scene graph analysis.")
            return

        # Activate object attributes
        for obj in inferred_scene_graph.get('objects', []):
            attributes = obj.get('attributes', {})
            for attr_name, attr_value in attributes.items():
                if attr_value and attr_value != 'unknown':
                    # Activate specific attribute value (e.g., 'circle', 'red')
                    self.slipnet.activate_node(attr_value, activation_value=0.6, is_initial=True)
                    # Also activate the general attribute type (e.g., 'shape', 'color')
                    if attr_name in ['shape', 'color', 'fill', 'size', 'orientation', 'texture']:
                        self.slipnet.activate_node(attr_name, activation_value=0.4, is_initial=True)

        # Activate relations
        for rel in inferred_scene_graph.get('relations', []):
            rel_type = rel.get('type')
            if rel_type and rel_type != 'none':
                self.slipnet.activate_node(rel_type, activation_value=0.8, is_initial=True)
        
        logger.debug("SymbolicEngine: Scene graph analysis complete, Slipnet nodes activated.")

    def update_workspace(self, 
                         query_sg_view1: Dict[str, Any],
                         query_sg_view2: Dict[str, Any],
                         support_sgs: List[Dict[str, Any]],
                         step: int):
        """
        Updates the internal workspace with the latest scene graphs and step.
        """
        self.workspace['query_scene_graph_view1'] = query_sg_view1
        self.workspace['query_scene_graph_view2'] = query_sg_view2
        self.workspace['support_scene_graphs'] = support_sgs
        self.workspace['step'] = step
        self.workspace['active_concepts'] = self.slipnet.get_active_nodes(
            threshold=self.config['slipnet_config']['activation_threshold']
        )
        logger.debug(f"SymbolicEngine: Workspace updated at step {step}.")

    def run_symbolic_reasoning_step(self, current_step: int):
        """
        Executes one step of symbolic reasoning.
        This involves:
        1. Propagating Slipnet activations.
        2. Scheduling and running Codelets (via Coderack, not directly here).
        3. RuleInducer proposes/tests rules based on active concepts.
        4. Workspace is updated.
        """
        logger.debug(f"SymbolicEngine: Running reasoning step {current_step}.")

        # Step 1: Propagate Slipnet activations
        self.slipnet.step(
            decay_factor=self.config['slipnet_config']['general_decay_factor'],
            max_activation=self.config['slipnet_config']['max_activation']
        )

        # Step 2: Interact with RuleInducer (via Codelets in a real Copycat system)
        # For this simplified integration, we'll directly call RuleInducer methods.
        # In a full Copycat, Codelets would be scheduled on a Coderack, and they would
        # call these RuleInducer methods based on their urgency.

        # RuleInducer proposes new rule fragments based on currently active Slipnet concepts
        proposed_rule_fragments = self.rule_inducer.propose_rule_fragments(self.workspace)
        self.workspace['current_rule_fragments'].extend(proposed_rule_fragments)

        # RuleInducer tests existing rule fragments
        # This would involve evaluating them against the current problem (query + support)
        # and potentially updating their "goodness" or "plausibility".
        # Successful rules might reinforce concepts in Slipnet.
        self.rule_inducer.test_rule_fragments(self.workspace)
        
        # Filter out low-confidence or redundant rule fragments
        self.workspace['current_rule_fragments'] = [
            rf for rf in self.workspace['current_rule_fragments'] 
            if rf.get('confidence', 0.0) > 0.1 # Example threshold
        ]

        # Log active concepts in the workspace for dashboard visualization
        self.workspace['active_concepts'] = self.slipnet.get_active_nodes(
            threshold=self.config['slipnet_config']['activation_threshold']
        )
        self.workspace['step'] = current_step
        
        logger.debug(f"SymbolicEngine: Reasoning step {current_step} complete. Active concepts: {self.workspace['active_concepts']}")
        logger.debug(f"SymbolicEngine: Current rule fragments: {len(self.workspace['current_rule_fragments'])}")

    def get_workspace_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current workspace state for logging/visualization."""
        return {
            'query_scene_graph_view1': self.workspace['query_scene_graph_view1'],
            'query_scene_graph_view2': self.workspace['query_scene_graph_view2'],
            'support_scene_graphs': self.workspace['support_scene_graphs'],
            'current_rule_fragments': [rf.to_dict() if hasattr(rf, 'to_dict') else rf for rf in self.workspace['current_rule_fragments']],
            'active_concepts': self.workspace['active_concepts'],
            'step': self.workspace['step']
        }

    def get_slipnet_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current Slipnet state for logging/visualization."""
        return self.slipnet.to_dict()

