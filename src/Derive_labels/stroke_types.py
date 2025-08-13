"""
stroke_types.py
Symbolic, compositional, and context-aware concept extraction for Bongard Solver.
All geometric/statistical logic, legacy code, and logging removed. Only symbolic concept extraction functions remain.
"""

def extract_symbolic_stroke_concepts(action_sequence, problem_context=None):
    """
    Extract symbolic stroke-level concepts from a LOGO action sequence.
    Args:
        action_sequence (list): List of LOGO action commands.
        problem_context (dict, optional): Context for concept extraction.
    Returns:
        dict: Symbolic stroke concepts (e.g., stroke type, compositional structure, context tags).
    """
    return {
        'stroke_types': get_stroke_types(action_sequence),
        'compositional_structure': analyze_stroke_structure(action_sequence),
        'context_tags': extract_context_tags(action_sequence, problem_context)
    }

def get_stroke_types(action_sequence):
    types = []
    for cmd in action_sequence:
        if 'line' in str(cmd):
            types.append('line')
        elif 'arc' in str(cmd):
            types.append('arc')
        else:
            types.append('unknown')
    return types

def analyze_stroke_structure(action_sequence):
    return {
        'sequence_patterns': find_repeating_stroke_patterns(action_sequence),
        'hierarchical_structure': build_stroke_tree(action_sequence),
        'compositional_rules': extract_stroke_composition_rules(action_sequence)
    }

def find_repeating_stroke_patterns(action_sequence):
    patterns = []
    seen = set()
    for cmd in action_sequence:
        if cmd in seen:
            patterns.append(cmd)
        else:
            seen.add(cmd)
    return patterns

def build_stroke_tree(action_sequence):
    return {'root': action_sequence[0] if action_sequence else None, 'children': action_sequence[1:]}

def extract_stroke_composition_rules(action_sequence):
    return [f'rule_{i}' for i, _ in enumerate(action_sequence)]

def extract_context_tags(action_sequence, problem_context=None):
    tags = []
    if problem_context:
        tags.extend(problem_context.get('tags', []))
    if any('zigzag' in str(cmd) for cmd in action_sequence):
        tags.append('pattern_zigzag')
    if any('circle' in str(cmd) for cmd in action_sequence):
        tags.append('shape_circle')
    return tags