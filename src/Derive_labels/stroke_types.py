"""
stroke_types.py
Symbolic, compositional, and context-aware concept extraction for Bongard Solver.
All geometric/statistical logic, legacy code, and logging removed. Only symbolic concept extraction functions remain.
"""

import re

def extract_modifier_from_stroke(action_command: str) -> dict:
    """
    Parses a LOGO shape action command into stroke class, modifier, and numerical params.
    Examples:
     - line_normal_1.000-0.833
     - arc_normal_0.500_0.542-0.750
    Returns a dict with:
      - class: 'line' or 'arc'
      - modifier: e.g., 'normal', 'triangle', 'circle', etc.
      - params: list of floats (length/angle or arc parameters)
    """
    # Pattern for line and arc commands
    pattern = r'^(line|arc)_(normal|triangle|circle|square|zigzag)_(.+)$'
    match = re.match(pattern, action_command)
    if not match:
        raise ValueError(f"Invalid action command format: {action_command}")
    stroke_class = match.group(1)
    modifier = match.group(2)
    numeric_part = match.group(3)
    if stroke_class == 'line':
        # Expected format: <float>-<float>
        parts = numeric_part.split('-')
        if len(parts) != 2:
            raise ValueError(f"Unexpected line numeric format: {numeric_part}")
        length = float(parts[0])
        angle = float(parts[1])
        params = [length, angle]
    elif stroke_class == 'arc':
        # Expected format: <float>_<float>-<float>
        parts_underscore = numeric_part.split('_')
        if len(parts_underscore) != 3:
            raise ValueError(f"Unexpected arc numeric format: {numeric_part}")
        param1 = float(parts_underscore[0])
        param2_str, param3_str = parts_underscore[1].split('-')
        param2 = float(param2_str)
        param3 = float(param3_str)
        params = [param1, param2, param3]
    else:
        raise ValueError(f"Unknown stroke class: {stroke_class}")
    return {
        'class': stroke_class,
        'modifier': modifier,
        'params': params
    }

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