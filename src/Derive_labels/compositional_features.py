import logging
from typing import Dict, List, Any, Optional
from src.physics_inference import PhysicsInference
from src.Derive_labels.shape_utils import _calculate_homogeneity, _calculate_pattern_regularity_from_modifiers
from src.Derive_labels.stroke_types import _extract_modifier_from_stroke, _calculate_stroke_specific_features, _calculate_stroke_type_differentiated_features
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage
from typing import List, Dict, Any


def _calculate_composition_features(action_commands: List[str]) -> Dict[str, Any]:
        """
        Calculate features about stroke composition and relationships.
        Expects a list of action command strings (not objects).
        Defensive: always convert to strings before any operation.
        """
        logger = logging.getLogger(__name__)
        strokes = ensure_all_strings(action_commands)
        logger.debug(f"[_calculate_composition_features] INPUTS: strokes count={len(strokes) if strokes else 0}")
        if not strokes:
            logger.debug("[_calculate_composition_features] No strokes, returning empty dict")
            return {}
        try:
            stroke_types = {}
            shape_modifiers = {}
            modifier_sequence = []
            for stroke in strokes:
                # Defensive: parse type and modifier from string
                if isinstance(stroke, str):
                    parts = stroke.split('_')
                    stroke_type = parts[0] if parts else 'unknown'
                    modifier = parts[1] if len(parts) > 1 else 'normal'
                else:
                    stroke_type = type(stroke).__name__.replace('Action', '').lower()
                    modifier = _extract_modifier_from_stroke(stroke)
                stroke_types[stroke_type] = stroke_types.get(stroke_type, 0) + 1
                shape_modifiers[modifier] = shape_modifiers.get(modifier, 0) + 1
                modifier_sequence.append(modifier)
            # Use json.dumps for serialization of distributions
            features = {
                'stroke_type_distribution': stroke_types,
                'shape_modifier_distribution': shape_modifiers,
                'stroke_diversity': len(stroke_types),
                'shape_diversity': len(shape_modifiers),
                'dominant_stroke_type': max(stroke_types.items(), key=lambda x: x[1])[0] if stroke_types else 'unknown',
                'dominant_shape_modifier': max(shape_modifiers.items(), key=lambda x: x[1])[0] if shape_modifiers else 'unknown'
            }
            features.update({
                'composition_complexity': len(strokes) + len(shape_modifiers),
                'homogeneity_score': _calculate_homogeneity(shape_modifiers),
                'pattern_regularity': _calculate_pattern_regularity_from_modifiers(modifier_sequence)
            })
            logger.debug(f"[_calculate_composition_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.warning(f"[_calculate_composition_features] Error: {e}")
            return {}

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