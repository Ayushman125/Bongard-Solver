# --- Symbolic, compositional, and context-aware concept extraction functions ---
from typing import List
def extract_symbolic_concepts_from_actions(action_sequence):
    """
    Extract abstract symbolic concepts from a LOGO action sequence.
    Returns a dict of high-level concept attributes (e.g., convexity, symmetry, containment).
    """
    concepts = {}
    # Example symbolic extraction logic
    concepts['convexity'] = detect_convex_pattern(action_sequence)
    concepts['symmetry'] = detect_symmetry_pattern(action_sequence)
    concepts['containment'] = detect_containment_pattern(action_sequence)
    concepts['stroke_types'] = extract_stroke_types(action_sequence)
    concepts['compositional_rules'] = extract_compositional_rules(action_sequence)
    return concepts

def extract_problem_level_features(positive_examples, negative_examples):
    """
    Extract discriminative features by comparing positive and negative examples at the problem level.
    Returns a list of consistent concept patterns.
    """
    discriminative_patterns = []
    for pos_actions, neg_actions in zip(positive_examples, negative_examples):
        pattern_diff = compare_action_patterns(pos_actions, neg_actions)
        discriminative_patterns.append(pattern_diff)
    return find_consistent_concept(discriminative_patterns)

def analyze_action_structure(action_sequence):
    """
    Analyze compositional structure of action programs.
    Returns a dict with sequence patterns, hierarchical structure, and compositional rules.
    """
    return {
        'sequence_patterns': find_repeating_patterns(action_sequence),
        'hierarchical_structure': build_action_tree(action_sequence),
        'compositional_rules': extract_compositional_rules(action_sequence)
    }

# --- Symbolic concept extraction helpers (stubs, to be implemented as needed) ---
def detect_convex_pattern(action_sequence):
    # Placeholder: implement symbolic convexity detection
    return 'abstract_convex' if 'convex' in str(action_sequence) else 'not_convex'

def detect_symmetry_pattern(action_sequence):
    # Placeholder: implement symbolic symmetry detection
    return 'symmetric' if 'symmetry' in str(action_sequence) else 'not_symmetric'

def detect_containment_pattern(action_sequence):
    # Placeholder: implement symbolic containment detection
    return 'contains' if 'contain' in str(action_sequence) else 'not_contained'

def extract_stroke_types(action_sequence):
    # Placeholder: extract stroke types from action sequence
    return ['line' if 'line' in str(action_sequence) else 'arc']

def extract_compositional_rules(action_sequence):
    # Placeholder: extract compositional rules from action sequence
    return ['rule1', 'rule2']

def compare_action_patterns(pos_actions, neg_actions):
    # Placeholder: compare positive and negative action patterns
    return {'difference': str(pos_actions) + ' vs ' + str(neg_actions)}

def find_consistent_concept(discriminative_patterns):
    # Placeholder: find consistent concept from patterns
    return {'consistent_concept': discriminative_patterns}

def find_repeating_patterns(action_sequence):
    # Placeholder: find repeating patterns in action sequence
    return ['repeat1', 'repeat2']

def build_action_tree(action_sequence):
    # Placeholder: build hierarchical tree from action sequence

    return {'tree': 'hierarchy'}

def open_stroke_convexity(vertices: List[tuple]) -> float:
    """Convexity for open polylines: count sign changes in turn angles."""
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    if not vertices or len(vertices) < 3:
        logger.info("Convexity: <3 vertices, returning 0.0 (degenerate)")
        return 0.0
    arr = np.array(vertices)
    from shapely.geometry import MultiPoint
    try:
        hull = MultiPoint(arr).convex_hull
        hull_area = hull.area if hasattr(hull, 'area') else 0.0
        poly_area = 0.0
        if len(vertices) >= 3:
            from shapely.geometry import Polygon
            poly = Polygon(vertices)
            poly_area = poly.area if poly.is_valid else 0.0
        if hull_area > 0 and poly_area > 0:
            ratio = float(poly_area / hull_area)
            logger.info(f"Convexity: poly_area={poly_area:.4f}, hull_area={hull_area:.4f}, ratio={ratio:.4f}")
            return ratio
        else:
            logger.info(f"Convexity: hull_area={hull_area:.4f}, poly_area={poly_area:.4f}, returning 0.0 (degenerate or fallback)")
            return 0.0

    except Exception as e:
        logger.error(f"Convexity: Exception occurred - {e}")
        return 0.0
