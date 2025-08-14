import numpy as np
def simulate_simplicity(vertices, min_dist=1e-2, perturb_scale=1e-1):
    """
    Perturb collinear or degenerate points to avoid geometric issues in shape rendering.
    This is a minimal implementation; you can improve it for your needs.
    """
    arr = np.array(vertices)
    if arr.shape[0] < 3:
        return vertices
    # Add small noise to each vertex
    noise = np.random.normal(0, 1e-3, arr.shape)
    arr = arr + noise
    return [tuple(v) for v in arr]

import math
import numpy as np
from typing import List, Tuple, Dict
from shapely.geometry import MultiPoint, Polygon

def extract_shape_vertices(action_commands: List[str]) -> List[Tuple[float, float]]:
    """
    Replay LOGO-style action commands and return a list of (x, y) vertices.
    Supports 'line' and 'arc' commands with normalized params.
    """
    vertices = []
    x, y = 0.0, 0.0
    heading = 0.0
    vertices.append((x, y))
    for cmd in action_commands:
        parts = cmd.split('_')
        if len(parts) < 2:
            continue
        stroke_class = parts[0]
        modifier = parts[1]
        param_str = '_'.join(parts[2:]) if len(parts) > 2 else '0.5-0.5'
        param_parts = param_str.split('-')
        if stroke_class == 'line':
            length = float(param_parts[0]) if param_parts else 0.5
            angle = float(param_parts[1]) if len(param_parts) > 1 else 0.0
            rad = math.radians(heading)
            dx = length * math.cos(rad)
            dy = length * math.sin(rad)
            x += dx
            y += dy
            vertices.append((x, y))
            heading += angle * 360.0
        elif stroke_class == 'arc':
            # Parse arc parameters, interpolate between start and end radius if two values are present
            if param_parts:
                if '_' in param_parts[0]:
                    radius_parts = param_parts[0].split('_')
                    try:
                        radius_start = float(radius_parts[0])
                        radius_end = float(radius_parts[1]) if len(radius_parts) > 1 else radius_start
                    except ValueError:
                        radius_start = radius_end = 0.5
                else:
                    try:
                        radius_start = radius_end = float(param_parts[0])
                    except ValueError:
                        radius_start = radius_end = 0.5
            else:
                radius_start = radius_end = 0.5
            span = float(param_parts[1]) if len(param_parts) > 1 else 0.25
            arc_points = 10
            for i in range(arc_points):
                # Interpolate radius for each arc point
                t = i / (arc_points - 1) if arc_points > 1 else 0
                radius_val = radius_start + t * (radius_end - radius_start)
                theta = heading + (span * 360.0) * t
                rad = math.radians(theta)
                px = x + radius_val * math.cos(rad)
                py = y + radius_val * math.sin(rad)
                vertices.append((px, py))
            heading += span * 360.0
            x, y = vertices[-1]
    return vertices
    def extract_topological_features(action_sequence: List[str]) -> np.ndarray:
        features = []
        for cmd in action_sequence:
            parts = cmd.split('_')
            if len(parts) < 2:
                features.extend([0, 0, 0])
                continue
            stroke_class = parts[0]
            param_str = '_'.join(parts[2:]) if len(parts) > 2 else '0.5-0.5'
            param_parts = param_str.split('-')
            if stroke_class == 'arc':
                # Robustly handle multi-parameter arcs, always produce three values
                if param_parts:
                    if '_' in param_parts[0]:
                        radius_parts = param_parts[0].split('_')
                        try:
                            radius_start = float(radius_parts[0])
                            radius_end = float(radius_parts[1]) if len(radius_parts) > 1 else radius_start
                        except ValueError:
                            radius_start = radius_end = 0.5
                    else:
                        try:
                            radius_start = radius_end = float(param_parts[0])
                        except ValueError:
                            radius_start = radius_end = 0.5
                else:
                    radius_start = radius_end = 0.5
                try:
                    span = float(param_parts[1]) if len(param_parts) > 1 else 0.25
                except ValueError:
                    span = 0.25
                features.extend([radius_start, radius_end, span])
            else:
                # For lines and others, always produce two values, pad with zero for consistency
                try:
                    val1 = float(param_parts[0]) if param_parts else 0.5
                except ValueError:
                    val1 = 0.5
                try:
                    val2 = float(param_parts[1]) if len(param_parts) > 1 else 0.0
                except ValueError:
                    val2 = 0.0
                features.extend([val1, val2, 0])
        return np.array(features)

def extract_symbolic_concepts_from_actions(action_sequence: List[str]) -> Dict:
    """
    Extract abstract symbolic concepts from a LOGO action sequence using geometric and graph-based analysis.
    Returns a dict of high-level concept attributes (convexity, symmetry, containment, stroke types, compositional rules).
    """
    vertices = extract_shape_vertices(action_sequence)
    concepts = {}
    concepts['convexity'] = compute_convexity(vertices)
    concepts['symmetry'] = compute_symmetry(vertices)
    concepts['containment'] = compute_containment(vertices)
    concepts['stroke_types'] = extract_stroke_types(action_sequence)
    concepts['compositional_rules'] = extract_compositional_rules(action_sequence)

    # Add robust action signature parsing for logging and feature extraction
    action_signatures = []
    for cmd in action_sequence:
        parts = cmd.split('_')
        if len(parts) < 2:
            # For malformed commands, pad with zeros for arc, or (0,0) for line
            action_signatures.append((0.0, 0.0, 0.0))
            continue
        stroke_class = parts[0]
        param_str = '_'.join(parts[2:]) if len(parts) > 2 else '0.5-0.5'
        param_parts = param_str.split('-')
        if stroke_class == 'arc':
            # Always return a tuple of three values for arc commands
            if param_parts:
                if '_' in param_parts[0]:
                    radius_parts = param_parts[0].split('_')
                    try:
                        radius_start = float(radius_parts[0])
                        radius_end = float(radius_parts[1]) if len(radius_parts) > 1 else radius_start
                    except ValueError:
                        radius_start = radius_end = 0.5
                else:
                    try:
                        radius_start = radius_end = float(param_parts[0])
                    except ValueError:
                        radius_start = radius_end = 0.5
            else:
                radius_start = radius_end = 0.5
            try:
                span = float(param_parts[1]) if len(param_parts) > 1 else 0.25
            except ValueError:
                span = 0.25
            # Use tuple of (radius_start, radius_end, span) for signature
            action_signatures.append((radius_start, radius_end, span))
        else:
            # For lines and others, always return a tuple of two values
            try:
                val1 = float(param_parts[0]) if param_parts else 0.5
            except ValueError:
                val1 = 0.5
            try:
                val2 = float(param_parts[1]) if len(param_parts) > 1 else 0.0
            except ValueError:
                val2 = 0.0
            action_signatures.append((val1, val2))
    concepts['action_signatures'] = action_signatures
    return concepts

def extract_problem_level_features(positive_examples: List[List[str]], negative_examples: List[List[str]]) -> List:
    """
    Extract discriminative features by comparing positive and negative examples at the problem level.
    Returns a list of consistent concept patterns.
    """
    discriminative_patterns = []
    for pos_actions, neg_actions in zip(positive_examples, negative_examples):
        pos_concepts = extract_symbolic_concepts_from_actions(pos_actions)
        neg_concepts = extract_symbolic_concepts_from_actions(neg_actions)
        pattern_diff = {k: pos_concepts[k] for k in pos_concepts if pos_concepts[k] != neg_concepts.get(k)}
        discriminative_patterns.append(pattern_diff)
    return discriminative_patterns

def analyze_action_structure(action_sequence: List[str]) -> Dict:
    """
    Analyze compositional structure of action programs using pattern mining and hierarchical analysis.
    Returns a dict with sequence patterns, hierarchical structure, and compositional rules.
    """
    return {
        'sequence_patterns': find_repeating_patterns(action_sequence),
        'hierarchical_structure': build_action_tree(action_sequence),
        'compositional_rules': extract_compositional_rules(action_sequence)
    }

def compute_convexity(vertices: List[Tuple[float, float]]) -> float:
    if not vertices or len(vertices) < 3:
        return 0.0
    arr = np.array(vertices)
    hull = MultiPoint(arr).convex_hull
    hull_area = hull.area if hasattr(hull, 'area') else 0.0
    poly_area = 0.0
    if len(vertices) >= 3:
        poly = Polygon(vertices)
        poly_area = poly.area if poly.is_valid else 0.0
    if hull_area > 0 and poly_area > 0:
        return float(poly_area / hull_area)
    return 0.0

def compute_symmetry(vertices: List[Tuple[float, float]]) -> float:
    # Example: symmetry score by comparing mirrored halves
    if not vertices or len(vertices) < 3:
        return 0.0
    arr = np.array(vertices)
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[-mid:][::-1]
    if len(left) != len(right):
        return 0.0
    return float(np.mean(np.linalg.norm(left - right, axis=1)))

def compute_containment(vertices: List[Tuple[float, float]]) -> int:
    # Example: containment as area ratio threshold
    if not vertices or len(vertices) < 3:
        return 0
    poly = Polygon(vertices)
    return int(poly.is_valid and poly.area > 0)

def extract_stroke_types(action_sequence: List[str]) -> List[str]:
    types = set()
    for cmd in action_sequence:
        parts = cmd.split('_')
        if parts:
            types.add(parts[0])
    return list(types)

def extract_compositional_rules(action_sequence: List[str]) -> List[str]:
    # Example: mine frequent n-grams as compositional rules
    n = 2
    rules = set()
    for i in range(len(action_sequence) - n + 1):
        rule = tuple(action_sequence[i:i+n])
        rules.add(str(rule))
    return list(rules)

def find_repeating_patterns(action_sequence: List[str]) -> List[str]:
    # Example: find repeating subsequences
    from collections import Counter
    counts = Counter(action_sequence)
    return [k for k, v in counts.items() if v > 1]

def build_action_tree(action_sequence: List[str]) -> Dict:
    # Example: build a simple hierarchy by grouping by stroke type
    tree = {}
    for cmd in action_sequence:
        parts = cmd.split('_')
        if parts:
            tree.setdefault(parts[0], []).append(cmd)
    return tree
