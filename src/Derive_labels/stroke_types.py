def _calculate_stroke_specific_features(stroke, stroke_index, context=None, shape_info=None):
    """
    Extracts rich, context-aware features from a single stroke.
    Args:
        stroke: The stroke object or action command string.
        stroke_index: Index of the stroke in the shape sequence.
        context: Optional, contextual info for dynamic adaptation.
        shape_info: Additional geometric info about the shape.
    Returns:
        features: dict with symbolic and continuous stroke features.
    """
    # Parse stroke class and modifier
    try:
        parsed = extract_modifier_from_stroke(stroke)
    except Exception as e:
        parsed = {'class': 'unknown', 'modifier': 'unknown', 'params': [0.0, 0.0]}
    stroke_class = parsed['class']
    modifier = parsed['modifier']
    params = parsed['params']
    # Geometric features (stub: replace with real geometric extraction as needed)
    length = params[0] if params else 0.0
    angle = params[1] if len(params) > 1 else 0.0
    curvature = params[2] if len(params) > 2 else 0.0
    # If shape_info provides vertices, use them for advanced metrics
    num_vertices = 0
    compactness = 0.0
    convexity_ratio = 0.0
    complexity = 0.0
    if shape_info and 'vertices' in shape_info:
        vertices = shape_info['vertices']
        num_vertices = len(vertices)
        # Example: use open_stroke_convexity from shape_utils
        try:
            from src.Derive_labels.shape_utils import open_stroke_convexity
            convexity_ratio = open_stroke_convexity(vertices)
        except Exception:
            convexity_ratio = 0.0
        # Example: compactness = perimeter^2 / area (stub)
        try:
            import numpy as np
            from shapely.geometry import Polygon
            poly = Polygon(vertices)
            perimeter = poly.length if poly.is_valid else 0.0
            area = poly.area if poly.is_valid else 1.0
            compactness = (perimeter ** 2) / area if area > 0 else 0.0
        except Exception:
            compactness = 0.0
        # Complexity: number of vertices as a proxy
        complexity = num_vertices
    features = {
        'stroke_index': stroke_index,
        'stroke_class': stroke_class,
        'modifier': modifier,
        'length': length,
        'angle': angle,
        'curvature': curvature,
        'compactness': compactness,
        'convexity_ratio': convexity_ratio,
        'complexity': complexity,
        'num_vertices': num_vertices,
    }
    # Optionally adapt features dynamically based on context (few-shot, analogy-making)
    if context is not None:
        # Example: context-driven feature adaptation (stub)
        if 'analogical_map' in context:
            features['modifier'] = context['analogical_map'].get(modifier, modifier)
    return features
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