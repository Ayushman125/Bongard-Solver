
"""
Spatial-Relational & Topological Feature Extraction Module
Symbolic, compositional, and context-aware concept extraction only. All geometric/statistical logic removed.
"""

def compute_spatial_topological_features(action_sequence, problem_context=None):
    """
    Symbolically compute spatial-relational and topological concepts for a Bongard image.
    Args:
        action_sequence (list): List of LOGO action commands.
        problem_context (dict, optional): Context for concept extraction.
    Returns:
        dict: Symbolic spatial/topological concepts.
    """
    return {
        'adjacency_pattern': detect_adjacency_pattern(action_sequence, problem_context),
        'intersection_pattern': detect_intersection_pattern(action_sequence, problem_context),
        'containment_pattern': detect_containment_pattern(action_sequence, problem_context),
        'compositional_structure': analyze_action_structure(action_sequence)
    }

def detect_adjacency_pattern(action_sequence, problem_context=None):
    # Example: Symbolically detect adjacency from sequence order or repeated types
    return 'adjacent' if any('adjacent' in str(cmd) for cmd in action_sequence) else 'not_adjacent'

def detect_intersection_pattern(action_sequence, problem_context=None):
    # Example: Symbolically detect intersection from action types
    return 'intersecting' if any('intersect' in str(cmd) for cmd in action_sequence) else 'not_intersecting'

def detect_containment_pattern(action_sequence, problem_context=None):
    # Example: Symbolically detect containment from nested or hierarchical commands
    return 'contains' if any('contain' in str(cmd) for cmd in action_sequence) else 'not_contains'

def analyze_action_structure(action_sequence):
    # Example: Analyze compositional structure of action programs
    return {
        'sequence_patterns': find_repeating_patterns(action_sequence),
        'hierarchical_structure': build_action_tree(action_sequence),
        'compositional_rules': extract_composition_rules(action_sequence)
    }

def find_repeating_patterns(action_sequence):
    patterns = []
    seen = set()
    for cmd in action_sequence:
        if cmd in seen:
            patterns.append(cmd)
        else:
            seen.add(cmd)
    return patterns

def build_action_tree(action_sequence):
    return {'root': action_sequence[0] if action_sequence else None, 'children': action_sequence[1:]}

def extract_composition_rules(action_sequence):
    return [f'rule_{i}' for i, _ in enumerate(action_sequence)]
