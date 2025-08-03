# ConceptNet whitelist of relation types to retain
CONCEPTNET_KEEP_RELS = {
    "PartOf", "HasA", "AtLocation", "MadeOf", "DerivedFrom",
    "RelatedTo", "IsA", "UsedFor", "CapableOf", "Desires",
    "HasProperty", "LocatedNear", "SimilarTo", "SymbolOf",
    "HasSubevent", "CausesDesire", "MotivatedByGoal", "ObstructedBy",
    "Causes", "HasPrerequisite", "HasFirstSubevent", "HasLastSubevent",
    "DefinedAs", "MannerOf", "ReceivesAction", "HasContext",
    "EntailsEvent", "InstanceOf", "ExternalURL", "EtymologicallyRelatedTo",
    "FormOf", "Antonym", "Synonym", "DistinctFrom", "NotUsedFor"
}


# Map raw labels (from NV-Logo metadata or motif miner) to normalized shapes
# UPDATED: Only support the 5 discovered Bongard-LOGO shape types + basic geometric primitives
SHAPE_MAP = {
    # === DISCOVERED BONGARD-LOGO SHAPE TYPES (5 total) ===
    "normal": "normal",     # 24,107 occurrences (48.7%) - straight lines, most common
    "circle": "circle",     # 6,256 occurrences (12.6%) - circular shapes/arcs
    "square": "square",     # 6,519 occurrences (13.2%) - square-based shapes
    "triangle": "triangle", # 5,837 occurrences (11.8%) - triangular shapes
    "zigzag": "zigzag",     # 6,729 occurrences (13.6%) - zigzag patterns
    
    # === BASIC GEOMETRIC PRIMITIVES (for action programs) ===
    "line": "line",         # Basic line primitive
    "arc": "arc",           # Basic arc primitive
    "point": "point",       # Basic point primitive
    "dot": "point",         # Alias for point
    
    # === LEGACY MAPPINGS (map to discovered types only) ===
    "tri": "triangle",
    "quad": "square",       # Map quadrilateral to square (closest discovered type)
    "rectangle": "square",  # Map rectangle to square (closest discovered type)
    "polygon": "normal",    # Map generic polygon to normal (most common type)
    "curve": "circle",      # Map curve to circle (closest curved type)
    "segment": "line",      # Map segment to line
    "vertex": "point",      # Map vertex to point
    
    # Remove all non-discovered shape types - they will be filtered out
}

# Commonsense label mapping for KB edge creation - UPDATED for discovered Bongard-LOGO shape types only
COMMONSENSE_LABEL_MAP = {
    # === DISCOVERED BONGARD-LOGO SHAPE TYPES ===
    "normal": "normal_line",     # Most common - straight line patterns
    "circle": "circle_shape",    # Circular/curved patterns
    "square": "square_shape",    # Square-based patterns
    "triangle": "triangle_shape", # Triangular patterns
    "zigzag": "zigzag_pattern",  # Zigzag/irregular patterns
    
    # === BASIC GEOMETRIC PRIMITIVES ===
    "line": "line_segment",
    "arc": "arc_segment",
    "point": "point_location",
    "dot": "point_location",
    
    # === STROKE TYPE MAPPINGS ===
    "line_normal": "normal_line_stroke",
    "line_square": "square_line_stroke",
    "line_triangle": "triangle_line_stroke",
    "line_circle": "circle_line_stroke",
    "line_zigzag": "zigzag_line_stroke",
    "arc_normal": "normal_arc_stroke",
    "arc_circle": "circle_arc_stroke",
    "arc_square": "square_arc_stroke",
    "arc_triangle": "triangle_arc_stroke",
    "arc_zigzag": "zigzag_arc_stroke",
    
    # === COMPOSITE PATTERNS ===
    "motif": "composite_pattern",
    "cluster": "grouped_elements",
    
    # Additional discovered shape mappings
    "detected_normal": "normal_line",
    "detected_circle": "circle_shape",
    "detected_square": "square_shape", 
    "detected_triangle": "triangle_shape",
    "detected_zigzag": "zigzag_pattern",
    "BongardShapeType.NORMAL": "normal_line",
    "BongardShapeType.CIRCLE": "circle_shape",
    "BongardShapeType.SQUARE": "square_shape",
    "BongardShapeType.TRIANGLE": "triangle_shape", 
    "BongardShapeType.ZIGZAG": "zigzag_pattern",
    
    # CRITICAL: Map the 5 discovered Bongard-LOGO shape types directly
    "normal": "normal",
    "circle": "circle", 
    "square": "square",
    "triangle": "triangle",
    "zigzag": "zigzag",
    
    # Stroke type mappings - preserve arc vs line distinction  
    "line_stroke": "line",
    "arc_stroke": "arc",  # Changed from "curve" to "arc" to preserve stroke type
    "polygon_shape": "polygon",
}

# Motif categories and semantic types (for supernodes)
MOTIF_CATEGORIES = [
    "sequence", "group", "grid", "enclosed", "symmetry", "repeat", "adjacent", "enclosure", "connects-to", "mirrored", "spiral", "reflection", "loop"
]

# Standard semantic fields for objects and motifs
SEMANTIC_FIELDS = [
    "object_id", "vertices", "object_type", "is_closed", "fallback_geometry", "semantic_label", "shape_label", "motif_label", "motif_type", "kb_concept", "pattern_role", "action_program_type", "function_label", "gnn_score", "vl_embed"
]

# LOGO geometric predicate registry for edge construction
import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString

BASIC_LOGO_PREDICATES = {
    'adjacent_endpoints': lambda a, b: (
        a.get('object_type') == 'line' and
        b.get('object_type') == 'line' and
        any(np.allclose(np.array(pt1), np.array(pt2), atol=1e-3)
            for pt1 in a.get('endpoints', [])
            for pt2 in b.get('endpoints', []))
    ),
    'length_sim': lambda a, b: (
        a.get('length') and b.get('length') and
        abs(a['length'] - b['length']) < 0.05 * max(a['length'], b['length'])
    ),
    'angle_sim':  lambda a, b: (
        a.get('orientation') is not None and
        b.get('orientation') is not None and
        abs(a['orientation'] - b['orientation']) < 5
    ),
}

# STATE-OF-THE-ART: Abstract predicates for BONGARD-LOGO style reasoning
# Updated to handle all 5 discovered shape types: normal, circle, square, triangle, zigzag
ABSTRACT_PREDICATES = {
    # Convexity-based predicates
    'is_convex': lambda a, b=None: _is_convex_shape(a),
    'convexity_similar': lambda a, b: _convexity_similar(a, b),
    
    # Symmetry-based predicates  
    'has_symmetry': lambda a, b=None: _has_symmetry(a),
    'symmetry_match': lambda a, b: _symmetry_types_match(a, b),
    
    # Count-based predicates (BONGARD-LOGO core concept)
    'same_line_count': lambda a, b: _same_stroke_count(a, b, 'line'),
    'same_arc_count': lambda a, b: _same_stroke_count(a, b, 'arc'),
    'same_intersection_count': lambda a, b: _same_intersection_count(a, b),
    
    # Shape type predicates for discovered Bongard-LOGO types
    'contains_normal_lines': lambda a, b=None: _contains_shape_type(a, 'normal'),
    'contains_circles': lambda a, b=None: _contains_shape_type(a, 'circle'),
    'contains_squares': lambda a, b=None: _contains_shape_type(a, 'square'),
    'contains_triangles': lambda a, b=None: _contains_shape_type(a, 'triangle'),
    'contains_zigzag': lambda a, b=None: _contains_shape_type(a, 'zigzag'),
    
    # Shape complexity predicates
    'is_simple_shape': lambda a, b=None: _get_shape_complexity(a) <= 1,
    'is_moderate_shape': lambda a, b=None: _get_shape_complexity(a) == 2,
    'is_complex_shape': lambda a, b=None: _get_shape_complexity(a) >= 3,
    
    # Topological predicates
    'has_holes': lambda a, b=None: _has_holes(a),
    'same_hole_count': lambda a, b: _same_hole_count(a, b),
    
    # Program-semantic predicates
    'same_action_sequence': lambda a, b: _same_action_sequence_type(a, b),
    'program_complexity_similar': lambda a, b: _program_complexity_similar(a, b),
    
    # Abstract geometric relations
    'forms_enclosure': lambda a, b: _forms_enclosure(a, b),
    'parallel_alignment': lambda a, b: _parallel_alignment(a, b),
    'nested_relation': lambda a, b: _nested_relation(a, b),
}

def _is_convex_shape(obj: dict) -> bool:
    """Check if a shape is convex"""
    vertices = obj.get('vertices', [])
    if len(vertices) < 3:
        return True  # Lines are convex
    
    try:
        polygon = Polygon(vertices)
        if polygon.is_valid:
            convex_hull = polygon.convex_hull
            convexity_ratio = polygon.area / convex_hull.area if convex_hull.area > 0 else 0
            return convexity_ratio > 0.95  # 95% threshold for convexity
        return False
    except Exception:
        return False

def _convexity_similar(obj_a: dict, obj_b: dict) -> bool:
    """Check if two objects have similar convexity"""
    conv_a = _calculate_convexity_score(obj_a)
    conv_b = _calculate_convexity_score(obj_b)
    return abs(conv_a - conv_b) < 0.1

def _calculate_convexity_score(obj: dict) -> float:
    """Calculate convexity score [0,1] where 1 is perfectly convex"""
    vertices = obj.get('vertices', [])
    if len(vertices) < 3:
        return 1.0
    
    try:
        polygon = Polygon(vertices)
        if polygon.is_valid:
            convex_hull = polygon.convex_hull
            return polygon.area / convex_hull.area if convex_hull.area > 0 else 0
        return 0.5
    except Exception:
        return 0.5

def _has_symmetry(obj: dict) -> bool:
    """Check if object has any significant symmetry"""
    vertices = obj.get('vertices', [])
    if len(vertices) < 3:
        return True  # Lines are symmetric
    
    try:
        # Calculate symmetry score using method from predicate_induction
        from src.scene_graphs_building.predicate_induction import _calculate_symmetry_score
        symmetry_score = _calculate_symmetry_score(vertices)
        return symmetry_score > 0.7  # 70% threshold for symmetry
    except Exception:
        return False

def _symmetry_types_match(obj_a: dict, obj_b: dict) -> bool:
    """Check if two objects have similar symmetry characteristics"""
    try:
        from src.scene_graphs_building.predicate_induction import _calculate_symmetry_score
        sym_a = _calculate_symmetry_score(obj_a.get('vertices', []))
        sym_b = _calculate_symmetry_score(obj_b.get('vertices', []))
        return abs(sym_a - sym_b) < 0.2
    except Exception:
        return False

def _same_stroke_count(obj_a: dict, obj_b: dict, stroke_type: str) -> bool:
    """Count strokes of specific type in action programs"""
    def count_strokes(obj, stype):
        action_program = obj.get('action_program', [])
        count = 0
        for action in action_program:
            action_str = str(action).lower()
            if stype in action_str:
                count += 1
        return count
    
    count_a = count_strokes(obj_a, stroke_type)
    count_b = count_strokes(obj_b, stroke_type)
    return count_a == count_b

def _same_intersection_count(obj_a: dict, obj_b: dict) -> bool:
    """Check if objects have same number of self-intersections"""
    try:
        from src.scene_graphs_building.predicate_induction import _count_self_intersections
        int_a = _count_self_intersections(obj_a.get('vertices', []))
        int_b = _count_self_intersections(obj_b.get('vertices', []))
        return int_a == int_b
    except Exception:
        return False

def _has_holes(obj: dict) -> bool:
    """Check if shape has holes or enclosures"""
    try:
        from src.scene_graphs_building.predicate_induction import _detect_holes_and_enclosures
        hole_count = _detect_holes_and_enclosures(obj.get('vertices', []))
        return hole_count > 0
    except Exception:
        return False

def _same_hole_count(obj_a: dict, obj_b: dict) -> bool:
    """Check if objects have same number of holes"""
    try:
        from src.scene_graphs_building.predicate_induction import _detect_holes_and_enclosures
        holes_a = _detect_holes_and_enclosures(obj_a.get('vertices', []))
        holes_b = _detect_holes_and_enclosures(obj_b.get('vertices', []))
        return holes_a == holes_b
    except Exception:
        return False

def _same_action_sequence_type(obj_a: dict, obj_b: dict) -> bool:
    """Check if objects have similar action sequence patterns"""
    def get_action_pattern(obj):
        action_program = obj.get('action_program', [])
        pattern = []
        for action in action_program:
            if 'line' in str(action).lower():
                pattern.append('L')
            elif 'arc' in str(action).lower():
                pattern.append('A')
            elif 'turn' in str(action).lower():
                pattern.append('T')
            else:
                pattern.append('O')
        return ''.join(pattern)
    
    pattern_a = get_action_pattern(obj_a)
    pattern_b = get_action_pattern(obj_b)
    return pattern_a == pattern_b

def _program_complexity_similar(obj_a: dict, obj_b: dict) -> bool:
    """Check if objects have similar program complexity"""
    try:
        from src.scene_graphs_building.predicate_induction import _calculate_stroke_complexity
        comp_a = _calculate_stroke_complexity(obj_a)
        comp_b = _calculate_stroke_complexity(obj_b)
        return abs(comp_a - comp_b) < 0.2
    except Exception:
        return False

def _forms_enclosure(obj_a: dict, obj_b: dict) -> bool:
    """Check if one object encloses the other"""
    # Implementation would require spatial analysis
    return False

def _contains_shape_type(obj: dict, shape_type: str) -> bool:
    """Check if object contains a specific shape type from the 5 discovered Bongard-LOGO types"""
    if obj is None:
        return False
    
    # Check shape_type field directly
    obj_shape_type = obj.get('shape_type', '').lower()
    if obj_shape_type == shape_type.lower():
        return True
    
    # Check action program for shape type indicators
    action_program = obj.get('action_program', [])
    for action in action_program:
        if isinstance(action, str) and shape_type.lower() in action.lower():
            return True
    
    # Check semantic shapes
    semantic_shapes = obj.get('semantic_shapes', [])
    for shape in semantic_shapes:
        if shape.get('type', '').lower() == shape_type.lower():
            return True
    
    return False

def _get_shape_complexity(obj: dict) -> int:
    """Get shape complexity level based on discovered Bongard-LOGO complexity categories"""
    if obj is None:
        return 0
    
    # Check if complexity is already computed
    complexity = obj.get('complexity_level')
    if complexity is not None:
        return complexity
    
    # Check shape-specific complexity
    shape_type = obj.get('shape_type', '').lower()
    shape_complexity_map = {
        'normal': 1,      # Simplest - straight lines
        'circle': 2,      # Simple geometric shape
        'square': 2,      # Simple geometric shape  
        'triangle': 2,    # Simple geometric shape
        'zigzag': 3       # Most complex - irregular pattern
    }
    
    if shape_type in shape_complexity_map:
        return shape_complexity_map[shape_type]
    
    # Fallback: estimate complexity from action program length
    action_program = obj.get('action_program', [])
    if len(action_program) <= 3:
        return 1  # Simple
    elif len(action_program) <= 8:
        return 2  # Moderate
    else:
        return 3  # Complex
    vertices_a = obj_a.get('vertices', [])
    vertices_b = obj_b.get('vertices', [])
    
    if len(vertices_a) < 3 or len(vertices_b) < 3:
        return False
    
    try:
        poly_a = Polygon(vertices_a)
        poly_b = Polygon(vertices_b)
        
        if poly_a.is_valid and poly_b.is_valid:
            return poly_a.contains(poly_b) or poly_b.contains(poly_a)
        return False
    except Exception:
        return False

def _parallel_alignment(obj_a: dict, obj_b: dict) -> bool:
    """Check if objects have parallel orientations"""
    orient_a = obj_a.get('orientation')
    orient_b = obj_b.get('orientation')
    
    if orient_a is None or orient_b is None:
        return False
    
    # Check for parallel orientations (within 5 degrees)
    angle_diff = abs(orient_a - orient_b) % 180
    return angle_diff < 5 or angle_diff > 175

def _nested_relation(obj_a: dict, obj_b: dict) -> bool:
    """Check for nested spatial relationship"""
    bbox_a = obj_a.get('bounding_box')
    bbox_b = obj_b.get('bounding_box')
    
    if not bbox_a or not bbox_b:
        return False
    
    # Check if one bounding box is entirely within the other
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    
    # A contains B
    contains_ab = (x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b)
    # B contains A  
    contains_ba = (x1b <= x1a and y1b <= y1a and x2b >= x2a and y2b >= y2a)
    
    return contains_ab or contains_ba
