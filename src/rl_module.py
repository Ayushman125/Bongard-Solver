# Folder: bongard_solver/src/rl_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

# Import BongardRule and RuleAtom from bongard_rules
# Assuming bongard_rules.py exists and defines these constants and classes
try:
    from bongard_rules import BongardRule, RuleAtom, ALL_RULE_ATOMS
except ImportError:
    logging.warning("Could not import bongard_rules. Using dummy BongardRule and RuleAtom.")
    class RuleAtom:
        def __init__(self, name, is_value=False, arity=0):
            self.name = name
            self.is_value = is_value
            self.arity = arity
        def __repr__(self): return self.name
    class BongardRule:
        def __init__(self, name, description, program_ast, logical_facts):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
        def __repr__(self): return self.name
    ALL_RULE_ATOMS = {
        "SHAPE": RuleAtom("SHAPE", arity=2), "COLOR": RuleAtom("COLOR", arity=2),
        "circle": RuleAtom("circle", is_value=True), "red": RuleAtom("red", is_value=True),
        "AND": RuleAtom("AND", arity=-1), "OR": RuleAtom("OR", arity=-1), "NOT": RuleAtom("NOT", arity=1),
        "FORALL": RuleAtom("FORALL", arity=2), "EXISTS": RuleAtom("EXISTS", arity=2),
        "object_variable": RuleAtom("object_variable", is_value=True),
        "IMPLIES": RuleAtom("IMPLIES", arity=2)
    }


# Import rule_evaluator for precise reward calculation
try:
    from rule_evaluator import evaluate_rule_on_support_set
except ImportError:
    logging.warning("Could not import rule_evaluator. Using dummy evaluate_rule_on_support_set.")
    def evaluate_rule_on_support_set(rule: BongardRule, scene_graphs: List[Dict[str, Any]], labels: List[int]) -> float:
        """Dummy rule evaluator: always returns 0.5."""
        logging.warning("Using dummy rule_evaluator. Reward will be constant.")
        return 0.5 # Dummy accuracy

# Import ASTToFactsTransducer from dsl for rule construction
try:
    from dsl import ASTToFactsTransducer, ASTNode, Primitive, DSLProgram, DSL_VALUES, DSL_FUNCTIONS
except ImportError:
    logging.warning("Could not import DSL components. Rule construction will be limited.")
    # Dummy classes if dsl.py is not fully accessible
    class ASTNode:
        def __init__(self, primitive, children=None, value=None): self.primitive = primitive; self.children = children or []; self.value = value
        def to_dict(self): return {"op": self.primitive.name, "args": [c.to_dict() for c in self.children]}
    class Primitive:
        def __init__(self, name, func, type_signature, is_terminal=False): self.name = name; self.func = func; self.type_signature = type_signature; self.is_terminal = is_terminal
    class DSLProgram:
        def __init__(self, root_node): self.root = root_node
    class ASTToFactsTransducer:
        def convert(self, program): return [f"DUMMY_FACT({program.root.primitive.name})"]
    DSL_VALUES = []
    DSL_FUNCTIONS = []


logger = logging.getLogger(__name__)

# --- Rule Graph Space (Action Space) ---
class RuleGraphSpace:
    """
    Defines the action space for proposing rule components (RuleAtoms or values).
    Each action corresponds to selecting a specific primitive from the DSL.
    """
    def __init__(self):
        # Map each primitive name to an index
        # We need to ensure ALL_RULE_ATOMS is a dictionary of RuleAtom objects
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
    def __init__(self, support_set_features: List[torch.Tensor],
                 support_set_labels: List[int],
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
        
        # Initialize ASTToFactsTransducer
        self.transducer = ASTToFactsTransducer()
        
        # Map RuleAtom names to DSL Primitive objects for AST construction
        self.primitive_map = {p.name: p for p in DSL_VALUES + DSL_FUNCTIONS}
        
        logger.info(f"BongardEnv initialized. Max steps: {max_steps}.")

    def reset(self, support_set_features: List[torch.Tensor],
              support_set_labels: List[int],
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

        # Heuristic for "rule complete": if we have enough components to form a basic rule, or max steps.
        # This is still a heuristic. A more robust system would have an explicit "end_rule" action
        # or a grammar-based completion check.
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

        # This parser is highly simplified. A full parser would need to handle
        # operator precedence, variable binding, and arbitrary nesting.
        # We will attempt to construct a few common rule patterns.

        try:
            # Pattern 1: Quantified Attribute Predicate (e.g., FORALL(O, shape(O, circle)))
            # Components: [QUANTIFIER, OBJ_VAR, ATTR_PREDICATE, OBJ_VAR, VALUE]
            if (len(components) >= 5 and
                components[0].name in ["FORALL", "EXISTS"] and
                components[1].name == "object_variable" and
                components[2].arity == 2 and not components[2].is_value and # e.g., SHAPE, COLOR
                components[3].name == "object_variable" and # Should be the same object variable
                components[4].is_value): # e.g., circle, red
                
                quantifier_atom = components[0]
                obj_var_atom = components[1]
                attribute_atom = components[2]
                value_atom = components[4]

                # Create AST Nodes
                value_node = ASTNode(self.primitive_map[value_atom.name], value=value_atom.name)
                obj_var_node = ASTNode(self.primitive_map[obj_var_atom.name], value="O") # Standardize to 'O'
                
                # Attribute predicate (e.g., shape(O, circle))
                predicate_node = ASTNode(self.primitive_map[attribute_atom.name], 
                                         children=[obj_var_node, value_node])
                
                # Quantified expression (e.g., FORALL(O, predicate_node))
                root_node = ASTNode(self.primitive_map[quantifier_atom.name], 
                                    children=[obj_var_node, predicate_node])
                
                program_ast_root = root_node.to_dict()
                program = DSLProgram(root_node)
                logical_facts = self.transducer.convert(program)
                
                return BongardRule(
                    name=f"Proposed_{quantifier_atom.name}_{attribute_atom.name}_{value_atom.name}",
                    description=f"RL proposed rule: {quantifier_atom.name}(O, {attribute_atom.name}(O, {value_atom.name}))",
                    program_ast=[program_ast_root],
                    logical_facts=logical_facts
                )

            # Pattern 2: Simple Logical AND/OR of two attribute predicates
            # Components: [LOGICAL_OP, ATTR_PRED1, OBJ_VAR1, VALUE1, ATTR_PRED2, OBJ_VAR2, VALUE2]
            # This is a very specific sequence, assuming distinct object variables for now.
            elif (len(components) >= 7 and
                  components[0].name in ["AND", "OR"] and
                  components[1].arity == 2 and not components[1].is_value and # ATTR_PRED1
                  components[2].name == "object_variable" and components[3].is_value and # OBJ_VAR1, VALUE1
                  components[4].arity == 2 and not components[4].is_value and # ATTR_PRED2
                  components[5].name == "object_variable" and components[6].is_value): # OBJ_VAR2, VALUE2
                
                logical_op_atom = components[0]
                pred1_atom = components[1]
                obj_var1_atom = components[2]
                value1_atom = components[3]
                pred2_atom = components[4]
                obj_var2_atom = components[5]
                value2_atom = components[6]

                # Create AST Nodes for predicates
                obj_node1 = ASTNode(self.primitive_map[obj_var1_atom.name], value="O1")
                value_node1 = ASTNode(self.primitive_map[value1_atom.name], value=value1_atom.name)
                predicate_node1 = ASTNode(self.primitive_map[pred1_atom.name], children=[obj_node1, value_node1])

                obj_node2 = ASTNode(self.primitive_map[obj_var2_atom.name], value="O2")
                value_node2 = ASTNode(self.primitive_map[value2_atom.name], value=value2_atom.name)
                predicate_node2 = ASTNode(self.primitive_map[pred2_atom.name], children=[obj_node2, value_node2])

                # Logical operation node
                root_node = ASTNode(self.primitive_map[logical_op_atom.name], 
                                    children=[predicate_node1, predicate_node2])
                
                program_ast_root = root_node.to_dict()
                program = DSLProgram(root_node)
                logical_facts = self.transducer.convert(program)

                return BongardRule(
                    name=f"Proposed_{logical_op_atom.name}_{pred1_atom.name}_{pred2_atom.name}",
                    description=f"RL proposed rule: {logical_op_atom.name}({pred1_atom.name}(O1, {value1_atom.name}), {pred2_atom.name}(O2, {value2_atom.name}))",
                    program_ast=[program_ast_root],
                    logical_facts=logical_facts
                )

            logger.debug(f"RL: Could not parse components into a known rule pattern: {components}")
            return None

        except Exception as e:
            logger.error(f"Error parsing rule components during AST construction: {e}. Components: {components}")
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

class RLAgent:
    """
    Reinforcement Learning Agent that learns to propose Bongard rules.
    Uses a policy gradient method (e.g., REINFORCE) to optimize rule generation.
    """
    class Solution:
        def __init__(self, score: float, description: str, rule: Optional[BongardRule] = None):
            self.score = score
            self.description = description
            self.rule = rule
        def __repr__(self):
            return f"Solution(score={self.score:.2f}, description='{self.description}')"

    def __init__(self, support_context_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99):
        """
        Args:
            support_context_dim (int): Dimensionality of the support set context feature.
            learning_rate (float): Learning rate for the policy optimizer.
            gamma (float): Discount factor for rewards.
        """
        self.env = None # Environment will be set when search is called
        self.policy = RulePolicy(support_context_dim, RuleGraphSpace().num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        logger.info(f"RLAgent initialized. LR: {learning_rate}, Gamma: {gamma}.")

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Selects an action based on the current state using the policy network.
        Args:
            state (torch.Tensor): The current observation (support set context).
        Returns:
            Tuple[int, torch.Tensor]: The chosen action index and its log probability.
        """
        state = state.unsqueeze(0) # Add batch dimension
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item(), m.log_prob(action)

    def learn(self):
        """
        Performs a policy update using collected rewards and log probabilities.
        """
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns (optional, but often helps stability)
        returns = torch.tensor(returns)
        if len(returns) > 1: # Avoid division by zero for single step episodes
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else: # If only one return, just use it as is
            pass # No normalization needed for single element

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs = []
        self.rewards = []
        logger.debug(f"RLAgent learned. Policy Loss: {loss.item():.4f}")

    def search(self, rules: List[str], support_set_features: List[torch.Tensor],
               support_set_labels: List[int], support_set_scene_graphs: List[Dict[str, Any]],
               num_episodes: int = 10) -> 'RLAgent.Solution':
        """
        Searches for the best Bongard rule using the RL agent.
        Args:
            rules (List[str]): Placeholder for rules from ILP/Causal (not directly used by RL here,
                               but represents the context of symbolic rules).
            support_set_features (List[torch.Tensor]): Feature vectors for support images.
            support_set_labels (List[int]): Labels for support images.
            support_set_scene_graphs (List[Dict[str, Any]]): Scene graphs for support images.
            num_episodes (int): Number of episodes (rule generation attempts) to run.
        Returns:
            RLAgent.Solution: The best rule found and its accuracy.
        """
        logger.info(f"RLAgent starting search for {num_episodes} episodes.")
        
        # Initialize the environment for the current problem
        self.env = BongardEnv(support_set_features, support_set_labels, support_set_scene_graphs)
        
        best_rule_accuracy = -1.0
        best_rule_solution: Optional[RLAgent.Solution] = None

        # Aggregate support set features into a single context vector for the policy
        if support_set_features:
            # Simple mean pooling of features for context
            support_context = torch.mean(torch.stack(support_set_features), dim=0)
        else:
            # Handle case with no features (e.g., return a zero tensor of appropriate dim)
            logger.warning("No support set features provided. Using zero vector as context.")
            support_context = torch.zeros(self.policy.fc1.in_features) # Match input dim of policy

        for episode in range(num_episodes):
            state = self.env.reset(support_set_features, support_set_labels, support_set_scene_graphs)
            episode_rewards = []
            
            logger.debug(f"Episode {episode + 1}/{num_episodes} started.")
            for t in range(self.env.max_steps):
                action_idx, log_prob = self.select_action(support_context)
                next_state, reward, done, info = self.env.step(action_idx)
                
                self.rewards.append(reward)
                episode_rewards.append(reward)

                if done:
                    # If a rule was proposed and evaluated
                    if info['rule_proposed']:
                        current_accuracy = info['rule_accuracy']
                        logger.debug(f"Episode {episode + 1} finished (step {t+1}). Rule: {info['rule_name']}, Accuracy: {current_accuracy:.4f}")
                        
                        if current_accuracy > best_rule_accuracy:
                            best_rule_accuracy = current_accuracy
                            best_rule_solution = RLAgent.Solution(
                                score=current_accuracy,
                                description=f"RL found rule: {info['rule_name']} (Acc: {current_accuracy:.4f})",
                                rule=info.get('proposed_rule_object') # If you pass the BongardRule object in info
                            )
                    else:
                        logger.debug(f"Episode {episode + 1} finished (step {t+1}). No valid rule proposed.")
                    break
            
            self.learn() # Update policy after each episode

        if best_rule_solution:
            logger.info(f"RLAgent search completed. Best rule accuracy: {best_rule_accuracy:.4f}")
            return best_rule_solution
        else:
            logger.warning("RLAgent search completed, but no valid rule was found or all rules had low accuracy.")
            return RLAgent.Solution(score=0.0, description="No valid rule found by RL agent.")

