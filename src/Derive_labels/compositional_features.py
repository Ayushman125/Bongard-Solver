import logging
from typing import Dict, List, Any, Optional
from src.physics_inference import symbolic_concept_features
from src.Derive_labels.stroke_types import extract_modifier_from_stroke, _calculate_stroke_specific_features, _calculate_stroke_type_differentiated_features
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage
from typing import List, Dict, Any


from src.Derive_labels.composition import CompositionEngine

def _calculate_composition_features(action_sequence, context=None):
    from src.Derive_labels.stroke_types import _calculate_stroke_specific_features
    primitives = []
    for idx, cmd in enumerate(action_sequence):
        # shape_info and context can be passed if available
        shape_info = context.get('shape_info') if context and 'shape_info' in context else None
        primitive = _calculate_stroke_specific_features(cmd, idx, context=context, shape_info=shape_info)
        primitives.append(primitive)
    # Optionally, embed primitives for neural modules here
    rules = CompositionEngine.learn_composition_rules(primitives, context)
    compositions = CompositionEngine.generate_combinations(primitives, rules)
    return {
        'primitives': primitives,
        'composition_rules': rules,
        'compositions': compositions
    }

def ensure_all_strings(items):
    """Ensure all items in the list are strings."""
    return [str(item) for item in items] if items else []

# --- Compositional Feature Functions ---
def hierarchical_clustering_heights(features: np.ndarray, method: str = 'ward') -> np.ndarray:
    """Compute hierarchical clustering tree heights."""
    Z = linkage(features, method)
    return Z[:, 2]

def composition_tree_depth(tree: Dict[Any, List[Any]], root: Any) -> int:
    """Compute depth of composition tree."""
    def _depth(node):
        if node not in tree or not tree[node]:
            return 1
        return 1 + max(_depth(child) for child in tree[node])
    return _depth(root)

def composition_tree_branching_factor(tree: Dict[Any, List[Any]], root: Any) -> float:
    """Compute average branching factor of tree."""
    nodes = [root]
    total_branches = 0
    total_nodes = 0
    while nodes:
        node = nodes.pop()
        children = tree.get(node, [])
        total_branches += len(children)
        total_nodes += 1
        nodes.extend(children)
    return total_branches / total_nodes if total_nodes else 0.0

def subgraph_isomorphism_frequencies(graphs: List[nx.Graph]) -> int:
    """Count isomorphic subgraphs among a list of graphs."""
    count = 0
    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            if i < j and nx.is_isomorphic(g1, g2):
                count += 1
    return count

def recursive_shape_patterns(shapes: List[Any], pattern_func) -> int:
    """Count repeated substructures by recursively searching for patterns."""
    return sum(pattern_func(shape) for shape in shapes)

def multi_level_symmetry_chains(symmetries: Dict[str, List[float]]) -> int:
    """Count symmetry chains at multiple levels."""
    return sum(len(chain) for chain in symmetries.values())

def layered_edge_complexity(edges: List[List[Any]]) -> float:
    """Measure edge complexity at different hierarchical levels."""
    return np.mean([len(e) for e in edges])

def overlapping_substructure_ratios(substructures: List[Any], overlap_func) -> float:
    """Compute ratio of overlapping area/length between substructures."""
    overlaps = [overlap_func(s1, s2) for i, s1 in enumerate(substructures) for j, s2 in enumerate(substructures) if i < j]
    return np.mean(overlaps) if overlaps else 0.0

def composition_regularity_score(parts: List[Any], position_func) -> float:
    """Quantify regularity in part arrangement."""
    positions = [position_func(p) for p in parts]
    return np.var(positions)

def nested_convex_hull_levels(groups: List[List[np.ndarray]]) -> int:
    """Compute convex hulls recursively for nested groups."""
    from scipy.spatial import ConvexHull
    levels = 0
    for group in groups:
        if len(group) > 2:
            hull = ConvexHull(np.vstack(group))
            levels += 1
    return levels