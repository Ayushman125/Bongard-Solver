# Folder: bongard_solver/
# File: slipnet.py

import logging
from typing import Dict, Any, List, Tuple, Optional
import collections

logger = logging.getLogger(__name__)

class ConceptNode:
    """
    Represents a single concept within the Slipnet.
    Concepts can be attributes (e.g., 'circle', 'red'), relations (e.g., 'left_of', 'same_size'),
    or higher-level abstract ideas (e.g., 'symmetry', 'group').
    """
    def __init__(self, name: str, depth: int, activation: float = 0.0, initial_activation_decay_rate: float = 0.05):
        """
        Initializes a ConceptNode.
        Args:
            name (str): The unique name of the concept (e.g., "circle", "red", "left_of").
            depth (int): The conceptual abstraction level (e.g., 0 for basic attributes,
                         higher for relations or abstract concepts).
            activation (float): The current activation level of the node (0.0 to 1.0).
            initial_activation_decay_rate (float): Rate at which initial activation decays.
        """
        self.name = name
        self.depth = depth
        self.activation = activation
        self.initial_activation_decay_rate = initial_activation_decay_rate
        self.links: Dict[str, Tuple['ConceptNode', float]] = {}  # {neighbor_name: (neighbor_node, weight)}
        self.incoming_activation = 0.0 # Accumulates activation from neighbors in a single step
        self.is_initial_activation = False # Flag for initial activation decay

    def add_link(self, neighbor_node: 'ConceptNode', weight: float):
        """
        Adds a directional link to a neighbor concept node.
        Args:
            neighbor_node (ConceptNode): The ConceptNode instance to link to.
            weight (float): The strength of the link (0.0 to 1.0).
        """
        self.links[neighbor_node.name] = (neighbor_node, weight)

    def spread_activation(self, decay_factor: float = 0.01):
        """
        Spreads activation from this node to its neighbors.
        Activation is spread proportionally to link weights.
        Args:
            decay_factor (float): General decay applied to activation during spread.
        """
        if self.activation <= 0:
            return

        # Decay current activation (represents concept "fading" if not reinforced)
        self.activation = max(0.0, self.activation - decay_factor)
        
        # Apply special decay for initial activations
        if self.is_initial_activation:
            self.activation = max(0.0, self.activation - self.initial_activation_decay_rate)
            if self.activation <= 0.01: # If activation is very low, it's no longer 'initial'
                self.is_initial_activation = False


        # Spread to neighbors
        total_link_weight = sum(weight for _, weight in self.links.values())
        if total_link_weight > 0:
            for neighbor_node, weight in self.links.values():
                # Activation spread is proportional to current activation and link strength
                spread_amount = self.activation * (weight / total_link_weight)
                neighbor_node.incoming_activation += spread_amount
                logger.debug(f"  {self.name} spreading {spread_amount:.4f} to {neighbor_node.name}")

    def apply_incoming_activation(self, max_activation: float = 1.0):
        """
        Applies accumulated incoming activation and resets the incoming buffer.
        Caps activation at max_activation.
        """
        self.activation = min(max_activation, self.activation + self.incoming_activation)
        self.incoming_activation = 0.0 # Reset for next step

    def set_activation(self, value: float, is_initial: bool = False):
        """Sets the activation of the node directly."""
        self.activation = max(0.0, min(1.0, value))
        self.is_initial_activation = is_initial

    def to_dict(self) -> Dict[str, Any]:
        """Converts the node to a dictionary for serialization/logging."""
        return {
            'name': self.name,
            'depth': self.depth,
            'activation': self.activation,
            'links': {neighbor_node.name: weight for neighbor_node, weight in self.links.values()}
        }


class Slipnet:
    """
    Manages the graph of interconnected ConceptNodes.
    Handles activation propagation and overall Slipnet state.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config['slipnet_config']
        self.nodes: Dict[str, ConceptNode] = {}
        self._initialize_slipnet_nodes()
        self._initialize_slipnet_links()
        logger.info("Slipnet initialized.")

    def _initialize_slipnet_nodes(self):
        """
        Initializes all predefined concept nodes based on attribute and relation maps.
        Adds basic attribute concepts (shape, color, size, etc.) and relation concepts.
        """
        # Basic Attributes (Depth 0)
        from config import ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP, \
                           ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP
        
        for attr_map in [ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP,
                         ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP]:
            for name in attr_map.keys():
                self.add_node(name, depth=0) # Basic attributes at depth 0

        # Relations (Depth 1)
        from config import RELATION_MAP
        for name in RELATION_MAP.keys():
            if name != 'none': # 'none' is usually a null relation, not a concept
                self.add_node(name, depth=1)

        # Abstract Concepts (Depth 2+) - Example
        self.add_node("group", depth=2)
        self.add_node("symmetry", depth=2)
        self.add_node("correspondence", depth=2)
        self.add_node("rule", depth=3)
        self.add_node("analogy", depth=3)
        self.add_node("problem", depth=3)

        logger.info(f"Slipnet: Initialized {len(self.nodes)} concept nodes.")

    def _initialize_slipnet_links(self):
        """
        Establishes predefined links between concept nodes with initial weights.
        These weights can be static or learned/adapted over time.
        """
        # Example links (customize extensively based on your domain knowledge)
        # Attribute-to-Attribute (e.g., 'red' related to 'color')
        from config import ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_SIZE_MAP, RELATION_MAP

        # Links from specific attribute values to their general attribute type
        for color_name in ATTRIBUTE_COLOR_MAP.keys():
            if color_name in self.nodes:
                if 'color' not in self.nodes: self.add_node('color', depth=1)
                self.add_link(color_name, 'color', 0.8)
        for shape_name in ATTRIBUTE_SHAPE_MAP.keys():
            if shape_name in self.nodes:
                if 'shape' not in self.nodes: self.add_node('shape', depth=1)
                self.add_link(shape_name, 'shape', 0.8)
        for size_name in ATTRIBUTE_SIZE_MAP.keys():
            if size_name in self.nodes:
                if 'size' not in self.nodes: self.add_node('size', depth=1)
                self.add_link(size_name, 'size', 0.8)
        
        # Attribute Type to Abstract Concepts
        if 'color' in self.nodes and 'group' in self.nodes: self.add_link('color', 'group', 0.5)
        if 'shape' in self.nodes and 'group' in self.nodes: self.add_link('shape', 'group', 0.6)
        if 'size' in self.nodes and 'group' in self.nodes: self.add_link('size', 'group', 0.4)

        # Relations to Abstract Concepts
        for rel_name in ['left_of', 'right_of', 'above', 'below']:
            if rel_name in self.nodes and 'correspondence' in self.nodes:
                self.add_link(rel_name, 'correspondence', 0.7)
        
        if 'same_shape' in self.nodes and 'symmetry' in self.nodes: self.add_link('same_shape', 'symmetry', 0.6)
        if 'same_color' in self.nodes and 'symmetry' in self.nodes: self.add_link('same_color', 'symmetry', 0.5)

        # Abstract Concepts to 'rule' or 'problem'
        if 'group' in self.nodes and 'rule' in self.nodes: self.add_link('group', 'rule', 0.7)
        if 'symmetry' in self.nodes and 'rule' in self.nodes: self.add_link('symmetry', 'rule', 0.8)
        if 'correspondence' in self.nodes and 'analogy' in self.nodes: self.add_link('correspondence', 'analogy', 0.8)
        if 'analogy' in self.nodes and 'problem' in self.nodes: self.add_link('analogy', 'problem', 0.9)


        logger.info(f"Slipnet: Initialized links between concept nodes.")

    def add_node(self, name: str, depth: int, activation: float = 0.0):
        """Adds a concept node to the Slipnet."""
        if name not in self.nodes:
            self.nodes[name] = ConceptNode(name, depth, activation, self.config['initial_activation_decay_rate'])
            logger.debug(f"Slipnet: Added node '{name}' (depth {depth}).")
        else:
            logger.debug(f"Slipnet: Node '{name}' already exists.")

    def add_link(self, src_name: str, dst_name: str, weight: float):
        """Adds a directed link between two existing concept nodes."""
        if src_name not in self.nodes:
            logger.warning(f"Slipnet: Source node '{src_name}' not found for link.")
            return
        if dst_name not in self.nodes:
            logger.warning(f"Slipnet: Destination node '{dst_name}' not found for link.")
            return
        
        self.nodes[src_name].add_link(self.nodes[dst_name], weight)
        logger.debug(f"Slipnet: Added link from '{src_name}' to '{dst_name}' with weight {weight}.")

    def activate_node(self, node_name: str, activation_value: float = 1.0, is_initial: bool = False):
        """
        Directly activates a concept node.
        Args:
            node_name (str): Name of the node to activate.
            activation_value (float): The value to set the activation to.
            is_initial (bool): If True, marks this as an initial activation
                               subject to faster decay.
        """
        if node_name in self.nodes:
            self.nodes[node_name].set_activation(activation_value, is_initial)
            logger.debug(f"Slipnet: Activated node '{node_name}' to {activation_value:.4f} (initial: {is_initial}).")
        else:
            logger.warning(f"Slipnet: Attempted to activate non-existent node '{node_name}'.")

    def step(self, decay_factor: float = 0.01, max_activation: float = 1.0):
        """
        Performs one step of activation propagation in the Slipnet.
        1. All nodes spread their current activation.
        2. All nodes apply their accumulated incoming activation.
        3. Activations are normalized.
        Args:
            decay_factor (float): General decay applied to activation during spread.
            max_activation (float): Maximum allowed activation for a node.
        """
        # Step 1: All nodes spread their activation and decay
        for node in self.nodes.values():
            node.spread_activation(decay_factor)

        # Step 2: All nodes apply incoming activation
        for node in self.nodes.values():
            node.apply_incoming_activation(max_activation)
        
        # Step 3: Normalize activations (optional, but good for stability)
        # Sum of all activations
        total_activation = sum(node.activation for node in self.nodes.values())
        if total_activation > 0:
            for node in self.nodes.values():
                node.activation /= total_activation # Normalize to sum to 1
        
        logger.debug(f"Slipnet: Stepped. Total activation: {total_activation:.4f}")

    def get_node_activation(self, node_name: str) -> float:
        """Returns the current activation of a specific node."""
        return self.nodes[node_name].activation if node_name in self.nodes else 0.0

    def get_active_nodes(self, threshold: float = 0.1) -> Dict[str, float]:
        """Returns a dictionary of active nodes and their activations above a threshold."""
        return {name: node.activation for name, node in self.nodes.items() if node.activation > threshold}

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire Slipnet to a dictionary for serialization/logging."""
        return {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'config': self.config
        }

