# src/emergent/concept_net.py

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConceptNode:
    """
    Represents a single concept within the Concept Network.
    Each node has a name, a conceptual depth, an activation level, and neighbors.
    """
    def __init__(self, name: str, depth: int):
        """
        Initializes a ConceptNode.
        Args:
            name (str): The name of the concept (e.g., 'shape', 'triangle', 'red').
            depth (int): Conceptual depth, influencing decay rate [1..100].
                         Higher depth means slower decay.
        """
        if not (1 <= depth <= 100):
            raise ValueError("ConceptNode depth must be between 1 and 100.")
        self.name = name
        self.depth = depth
        self.activation = 0.0        # Current activation level [0..1]
        self.neighbors: Dict[ConceptNode, float] = {}  # Neighbor node -> weight

    def decay(self):
        """
        Decreases the activation of the node based on its depth.
        Concepts with higher depth (more fundamental/abstract) decay slower.
        """
        # Decay factor: (1 - depth/100.0). Higher depth -> smaller decay factor -> slower decay
        self.activation *= (1 - self.depth / 100.0)
        self.activation = max(0.0, self.activation) # Ensure activation doesn't go below zero
        logger.debug(f"Concept '{self.name}' decayed to {self.activation:.4f}")

    def spread(self):
        """
        Spreads the current activation to neighboring nodes.
        The amount spread is proportional to the neighbor link weight.
        """
        current_activation = self.activation # Use current activation before it changes
        if current_activation > 0.01: # Only spread if there's significant activation
            for nbr_node, weight in self.neighbors.items():
                spread_amount = current_activation * weight
                nbr_node.activation = min(1.0, nbr_node.activation + spread_amount)
                logger.debug(f"Concept '{self.name}' spread {spread_amount:.4f} to '{nbr_node.name}' (new act: {nbr_node.activation:.4f})")

class ConceptNetwork:
    """
    Manages a network of interconnected concepts.
    It allows for activation, decay, and spreading of activation among concepts,
    simulating a basic cognitive process.
    """
    def __init__(self):
        """
        Initializes the ConceptNetwork with predefined nodes and links.
        """
        self.nodes: Dict[str, ConceptNode] = {} # name -> ConceptNode object
        self._init_nodes()
        self._init_links()
        logger.info("ConceptNetwork initialized with base nodes and links.")

    def _init_nodes(self):
        """
        Initializes a set of predefined concept nodes.
        These represent fundamental feature types and common feature values.
        """
        logger.debug("Initializing ConceptNetwork nodes...")
        # Feature types (higher depth as they are more abstract/fundamental)
        self.add_node('shape', depth=90)
        self.add_node('color', depth=90)
        self.add_node('size',  depth=80)
        self.add_node('position_h', depth=80) # Horizontal position
        self.add_node('position_v', depth=80) # Vertical position
        self.add_node('fill', depth=70) # e.g., filled, outlined
        self.add_node('orientation', depth=70) # e.g., horizontal, vertical, diagonal
        self.add_node('texture', depth=60) # e.g., smooth, rough

        # Feature values (lower depth as they are more specific/concrete)
        # Shapes
        for val in ['triangle', 'quadrilateral', 'circle', 'pentagon', 'hexagon']:
            self.add_node(val, depth=50)
        # Colors (example)
        for val in ['red', 'blue', 'green', 'yellow', 'black', 'white']:
            self.add_node(val, depth=50)
        # Fill types
        for val in ['filled', 'outlined']:
            self.add_node(val, depth=50)
        # Sizes
        for val in ['small', 'medium', 'large']:
            self.add_node(val, depth=50)
        # Horizontal positions
        for val in ['left', 'center_h', 'right']:
            self.add_node(val, depth=50)
        # Vertical positions
        for val in ['top', 'center_v', 'bottom']:
            self.add_node(val, depth=50)
        # Orientations
        for val in ['horizontal', 'vertical', 'diagonal_up', 'diagonal_down']:
            self.add_node(val, depth=50)
        # Textures
        for val in ['smooth', 'rough', 'patterned']:
            self.add_node(val, depth=50)
        logger.debug(f"Initialized {len(self.nodes)} concept nodes.")

    def _init_links(self):
        """
        Establishes initial links (connections) between concept nodes with weights.
        These links represent associations between concepts.
        """
        logger.debug("Initializing ConceptNetwork links...")
        # Link feature types to their values
        self.link('shape', 'triangle', 0.3)
        self.link('shape', 'quadrilateral', 0.3)
        self.link('shape', 'circle', 0.3)
        self.link('shape', 'pentagon', 0.3)
        self.link('shape', 'hexagon', 0.3)

        self.link('color', 'red', 0.3)
        self.link('color', 'blue', 0.3)
        self.link('color', 'green', 0.3)
        self.link('color', 'yellow', 0.3)
        self.link('color', 'black', 0.3)
        self.link('color', 'white', 0.3)

        self.link('fill', 'filled', 0.4)
        self.link('fill', 'outlined', 0.4)
        self.link('color', 'filled', 0.2) # Color can be related to fill

        self.link('size', 'small', 0.3)
        self.link('size', 'medium', 0.3)
        self.link('size', 'large', 0.3)

        self.link('position_h', 'left', 0.4)
        self.link('position_h', 'center_h', 0.4)
        self.link('position_h', 'right', 0.4)

        self.link('position_v', 'top', 0.4)
        self.link('position_v', 'center_v', 0.4)
        self.link('position_v', 'bottom', 0.4)

        self.link('orientation', 'horizontal', 0.3)
        self.link('orientation', 'vertical', 0.3)
        self.link('orientation', 'diagonal_up', 0.3)
        self.link('orientation', 'diagonal_down', 0.3)

        self.link('texture', 'smooth', 0.3)
        self.link('texture', 'rough', 0.3)
        self.link('texture', 'patterned', 0.3)

        # Example of cross-category links (e.g., 'small' might be weakly linked to 'circle' if often found together)
        self.link('small', 'circle', 0.1)
        self.link('large', 'quadrilateral', 0.1)
        logger.debug("Initialized ConceptNetwork links.")

    def add_node(self, name: str, depth: int):
        """
        Adds a new concept node to the network.
        Args:
            name (str): The unique name of the concept.
            depth (int): The conceptual depth of the node.
        Raises:
            ValueError: If a node with the same name already exists.
        """
        if name in self.nodes:
            logger.warning(f"Concept node '{name}' already exists. Skipping addition.")
            return
        self.nodes[name] = ConceptNode(name, depth)
        logger.debug(f"Added concept node: '{name}' (depth: {depth})")

    def link(self, name_a: str, name_b: str, weight: float):
        """
        Creates a bidirectional link between two concept nodes.
        Args:
            name_a (str): The name of the first concept.
            name_b (str): The name of the second concept.
            weight (float): The strength of the link (between 0 and 1).
        Raises:
            KeyError: If either concept name does not exist in the network.
            ValueError: If weight is not between 0 and 1.
        """
        if not (0 <= weight <= 1):
            raise ValueError("Link weight must be between 0 and 1.")
        
        node_a = self.nodes.get(name_a)
        node_b = self.nodes.get(name_b)

        if not node_a:
            logger.error(f"Concept node '{name_a}' not found for linking.")
            raise KeyError(f"Concept node '{name_a}' not found.")
        if not node_b:
            logger.error(f"Concept node '{name_b}' not found for linking.")
            raise KeyError(f"Concept node '{name_b}' not found.")

        node_a.neighbors[node_b] = weight
        node_b.neighbors[node_a] = weight # Bidirectional link
        logger.debug(f"Linked '{name_a}' and '{name_b}' with weight {weight}")

    def activate(self, name: str, amount: float):
        """
        Increases the activation of a specific concept node.
        Args:
            name (str): The name of the concept to activate.
            amount (float): The amount by which to increase activation.
        """
        node = self.nodes.get(name)
        if node:
            node.activation = min(1.0, node.activation + amount)
            logger.debug(f"Activated concept '{name}' by {amount:.4f} (new act: {node.activation:.4f})")
        else:
            logger.warning(f"Attempted to activate unknown concept: '{name}'")

    def get_activation(self, name: str) -> float:
        """
        Returns the current activation level of a concept node.
        Args:
            name (str): The name of the concept.
        Returns:
            float: The activation level (0.0 if concept not found).
        """
        node = self.nodes.get(name)
        if node:
            return node.activation
        else:
            logger.warning(f"Attempted to get activation for unknown concept: '{name}'. Returning 0.0.")
            return 0.0

    def decay_all(self):
        """
        Applies decay to the activation of all nodes in the network.
        """
        logger.debug("Decaying all concept activations...")
        for node in self.nodes.values():
            node.decay()

    def spread_all(self):
        """
        Spreads activation from all nodes to their neighbors.
        It's important to spread based on current activations before they change
        due to other spreads in the same cycle, so we iterate over a copy.
        """
        logger.debug("Spreading all concept activations...")
        # Create a copy of nodes to iterate, as activations will be modified
        for node in list(self.nodes.values()):
            node.spread()

