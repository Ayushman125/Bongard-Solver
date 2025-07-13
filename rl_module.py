# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple, Optional

# Import BongardRule and RuleAtom from bongard_rules
from bongard_rules import BongardRule, ALL_RULE_ATOMS

# Import rule_evaluator for precise reward calculation
from rule_evaluator import evaluate_rule_on_support_set

logger = logging.getLogger(__name__)

# --- Rule Graph Space (Action Space) ---
class RuleGraphSpace:
    """
    Defines the action space for proposing rule components (RuleAtoms or values).
    Each action corresponds to selecting a specific primitive from the DSL.
    """
    def __init__(self):
        # Map each primitive name to an index
        self.action_to_primitive = {idx: p for idx, p in enumerate(ALL_RULE_ATOMS.values())}
        self.primitive_to_action = {p: idx for idx, p in enumerate(ALL_RULE_ATOMS.values())}
        self.num_actions = len(self.action_to_primitive)
        logger.info(f"RuleGraphSpace initialized with {self.num_actions} actions.")

    def sample(self) -> int:
        """Samples a random action index."""
        return random.randint(0, self.num_actions - 1)

    def decode_action(self, action_idx: int) -> RuleAtom:
        """Decodes an action index into its corresponding RuleAtom."""
        if action_idx not in self.action_to_primitive:
            raise ValueError(f"Action index {action_idx} out of bounds for RuleGraphSpace.")
        return self.action_to_primitive[action_idx]


# --- Bongard RL Environment ---
class BongardEnv:
    """
    Reinforcement Learning environment for Bongard problems.
    The agent sequentially proposes components to construct a rule.
    Reward is given based on the constructed rule's correctness on the support set.
    """
    def __init__(self, support_set_features: List[torch.Tensor], support_set_labels: List[int],
                 support_set_scene_graphs: List[Dict[str, Any]], # Added for precise reward
                 max_steps: int = 50):
        """
        Args:
            support_set_features (List[torch.Tensor]): List of feature vectors for each image in the support set.
                                                        Each tensor is (D_feature,).
            support_set_labels (List[int]): List of labels (0 or 1) for each image in the support set.
            support_set_scene_graphs (List[Dict[str, Any]]): List of scene graph dictionaries for support images.
            max_steps (int): Maximum number of rule components to propose in an episode.
        """
        self.support_set_features = support_set_features
        self.support_set_labels = support_set_labels
        self.support_set_scene_graphs = support_set_scene_graphs
        self.max_steps = max_steps
        
        self.current_rule_components: List[RuleAtom] = [] # Stores sequence of proposed RuleAtoms
        self.current_step = 0
        self.done = False

        self.action_space = RuleGraphSpace()
        # Observation space is conceptually summarized by the support set context features
        self.observation_space_dim = support_set_features[0].shape[0] if support_set_features else 0
        logger.info(f"BongardEnv initialized. Max steps: {max_steps}.")

    def reset(self, support_set_features: List[torch.Tensor], support_set_labels: List[int],
              support_set_scene_graphs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Resets the environment for a new episode."""
        self.support_set_features = support_set_features
        self.support_set_labels = support_set_labels
        self.support_set_scene_graphs = support_set_scene_graphs
        self.current_rule_components = []
        self.current_step = 0
        self.done = False
        logger.debug("BongardEnv reset.")
        return self.support_set_features # Return initial observation (features of support set)

    def step(self, action_idx: int) -> Tuple[List[torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Takes an action (proposes a rule component) and returns next state, reward, done, info.
        """
        self.current_step += 1
        
        rule_component = self.action_space.decode_action(action_idx)
        self.current_rule_components.append(rule_component)

        reward = 0.0
        info = {'rule_proposed': False, 'rule_name': 'None', 'rule_accuracy': 0.0}

        # Check if episode is done (max steps reached or rule is "complete" - heuristic)
        # For simplicity, we'll consider a rule "complete" if it has a reasonable number of components
        # or if max_steps is reached. A more advanced system would have an explicit "end_rule" action.
        
        # Heuristic for "rule complete": if we have enough components to form a basic rule, or max steps.
        is_heuristic_rule_complete = len(self.current_rule_components) >= 3 and \
                                     (self.current_rule_components[-1].is_value or \
                                      self.current_rule_components[-1].name in ["AND", "OR", "NOT", "IMPLIES"])
        
        if self.current_step >= self.max_steps or is_heuristic_rule_complete:
            self.done = True
            
            proposed_rule = self._construct_rule_from_components(self.current_rule_components)
            
            if proposed_rule:
                info['rule_proposed'] = True
                info['rule_name'] = proposed_rule.name
                
                # Calculate precise reward using rule_evaluator
                accuracy = evaluate_rule_on_support_set(proposed_rule, self.support_set_scene_graphs, self.support_set_labels)
                reward = accuracy # Reward is the accuracy on the support set
                info['rule_accuracy'] = accuracy
            else:
                reward = -0.1 # Small penalty for invalid rule
                logger.debug("RL: Proposed rule was invalid.")
            
            logger.debug(f"RL Episode finished. Final reward: {reward:.2f}, Rule: {info['rule_name']}, Acc: {info['rule_accuracy']:.2f}")

        return self.support_set_features, reward, self.done, info

    def _construct_rule_from_components(self, components: List[RuleAtom]) -> Optional[BongardRule]:
        """
        Constructs a BongardRule object (with AST and logical facts) from a sequence of proposed RuleAtoms.
        This is a simplified, greedy parser for a subset of the DSL grammar.
        It attempts to build a valid AST.
        """
        if not components:
            return None

        # Stack-based parsing for a simplified grammar
        # Example: [FORALL, OBJ_VAR, COLOR, OBJ_VAR, RED] -> FORALL(O, color(O,red))
        
        # A more robust parser would use a formal grammar and parse tree.
        # This is a heuristic attempt to build a valid structure.
        
        # Let's assume the simplest case: a single quantified predicate
        # e.g., FORALL(O, shape(O, circle)) or EXISTS(O, color(O, red))
        # Or a simple logical combination: AND(shape(O1, circle), color(O1, red))
        
        # This parser is highly simplified. A full parser would need to handle
        # operator precedence, variable binding, and arbitrary nesting.
        
        try:
            # Try to parse a quantified predicate (FORALL/EXISTS)
            if len(components) >= 5 and components[0].name in ["FORALL", "EXISTS"] and \
               components[1].name == "object_variable" and \
               components[2].arity == 2 and not components[2].is_value and \
               components[3].name == "object_variable" and components[4].is_value:
                
                quantifier_op = components[0].name
                obj_var = components[1].name # 'object_variable'
                predicate_atom = components[2] # e.g., ATTR_COLOR
                predicate_obj_var = components[3].name # Should be same as obj_var
                value_atom = components[4] # e.g., COLOR_RED
                
                # Construct AST
                value_node_dict = {"op": value_atom.name}
                obj_var_node_dict = {"op": obj_var, "value": "O"} # Standardize to 'O'
                
                predicate_node_dict = {
                    "op": predicate_atom.name,
                    "args": [obj_var_node_dict, value_node_dict]
                }
                
                root_node_dict = {
                    "op": quantifier_op,
                    "args": [obj_var_node_dict, predicate_node_dict]
                }
                
                program_ast = [root_node_dict]
                
                # Convert to logical facts using the transducer
                from dsl import ASTToFactsTransducer
                transducer = ASTToFactsTransducer()
                logical_facts = transducer.convert(BongardRule(name="", description="", program_ast=program_ast))
                
                return BongardRule(
                    name=f"Proposed_{quantifier_op}_{predicate_atom.name}_{value_atom.name}",
                    description=f"RL proposed rule: {quantifier_op}(O, {predicate_atom.name}(O, {value_atom.name}))",
                    program_ast=program_ast,
                    logical_facts=logical_facts
                )

            # Try to parse a simple AND/OR of two predicates
            elif len(components) >= 6 and components[0].name in ["AND", "OR"] and \
                 components[1].arity == 2 and not components[1].is_value and \
                 components[2].name == "object_variable" and components[3].is_value and \
                 components[4].arity == 2 and not components[4].is_value and \
                 components[5].name == "object_variable" and components[6].is_value:
                
                logical_op = components[0].name
                pred1_atom = components[1]
                pred1_obj_var = components[2].name
                pred1_value_atom = components[3]
                pred2_atom = components[4]
                pred2_obj_var = components[5].name
                pred2_value_atom = components[6]

                obj_var_node_dict_1 = {"op": pred1_obj_var, "value": "O1"}
                obj_var_node_dict_2 = {"op": pred2_obj_var, "value": "O2"}

                pred1_node_dict = {
                    "op": pred1_atom.name,
                    "args": [obj_var_node_dict_1, {"op": pred1_value_atom.name}]
                }
                pred2_node_dict = {
                    "op": pred2_atom.name,
                    "args": [obj_var_node_dict_2, {"op": pred2_value_atom.name}]
                }
                root_node_dict = {
                    "op": logical_op,
                    "args": [pred1_node_dict, pred2_node_dict]
                }
                program_ast = [root_node_dict]

                from dsl import ASTToFactsTransducer
                transducer = ASTToFactsTransducer()
                logical_facts = transducer.convert(BongardRule(name="", description="", program_ast=program_ast))

                return BongardRule(
                    name=f"Proposed_{logical_op}_{pred1_atom.name}_{pred2_atom.name}",
                    description=f"RL proposed rule: {logical_op}({pred1_atom.name}(O1, {pred1_value_atom.name}), {pred2_atom.name}(O2, {pred2_value_atom.name}))",
                    program_ast=program_ast,
                    logical_facts=logical_facts
                )

            # If no simple pattern matches, return None
            logger.debug(f"RL: Could not parse components into a valid rule: {components}")
            return None

        except Exception as e:
            logger.error(f"Error parsing rule components: {e}. Components: {components}")
            return None


class RulePolicy(nn.Module):
    """
    Policy network for the RL agent, proposing rule components.
    Takes support set context and outputs probabilities over actions.
    """
    def __init__(self, support_context_dim: int, action_space_size: int):
        super().__init__()
        self.fc1 = nn.Linear(support_context_dim, 256)
        self.relu = nn.ReLU()
        self.fc_actions = nn.Linear(256, action_space_size) # Logits for actions
        logger.info(f"RulePolicy initialized. Context dim: {support_context_dim}, Action space: {action_space_size}.")

    def forward(self, support_set_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_set_context (torch.Tensor): Context vector from the support set (Batch_size, D_context).
                                                For RL, typically Batch_size=1 per episode.
        Returns:
            torch.Tensor: Logits for each possible action (Batch_size, Action_space_size).
        """
        x = self.fc1(support_set_context)
        x = self.relu(x)
        action_logits = self.fc_actions(x)
        return action_logits

