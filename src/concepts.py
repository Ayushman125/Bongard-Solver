# Problem-aware concept predicates for Bongard-Solver
# Updated to handle the 5 discovered Bongard-LOGO shape types: normal, circle, square, triangle, zigzag

# Shape type predicates based on discovered Bongard-LOGO types
def is_normal_shape(f):
    """Check if shape is normal type (straight lines, 24,107 occurrences)"""
    return f.get("shape_type", "") == "normal" or f.get("stroke_type", "") == "normal"

def is_circle_shape(f):
    """Check if shape is circle type (circular shapes/arcs, 6,256 occurrences)"""
    return f.get("shape_type", "") == "circle" or f.get("stroke_type", "") == "circle"

def is_square_shape(f):
    """Check if shape is square type (square-based shapes, 6,519 occurrences)"""
    return f.get("shape_type", "") == "square" or f.get("stroke_type", "") == "square"

def is_triangle_shape(f):
    """Check if shape is triangle type (triangular shapes, 5,837 occurrences)"""
    return f.get("shape_type", "") == "triangle" or f.get("stroke_type", "") == "triangle"

def is_zigzag_shape(f):
    """Check if shape is zigzag type (zigzag patterns, 6,729 occurrences)"""
    return f.get("shape_type", "") == "zigzag" or f.get("stroke_type", "") == "zigzag"

# Geometric complexity predicates for shape types
def has_regular_geometry(f):
    """Check if shape has regular geometry (square, triangle, circle)"""
    shape_type = f.get("shape_type", f.get("stroke_type", ""))
    return shape_type in ["square", "triangle", "circle"]

def has_irregular_geometry(f):
    """Check if shape has irregular geometry (normal, zigzag)"""
    shape_type = f.get("shape_type", f.get("stroke_type", ""))
    return shape_type in ["normal", "zigzag"]

def has_curved_geometry(f):
    """Check if shape has curved geometry (circle, zigzag)"""
    shape_type = f.get("shape_type", f.get("stroke_type", ""))
    return shape_type in ["circle", "zigzag"] or f.get("is_highly_curved", False)

def has_straight_geometry(f):
    """Check if shape has straight geometry (normal, square, triangle)"""
    shape_type = f.get("shape_type", f.get("stroke_type", ""))
    return shape_type in ["normal", "square", "triangle"]

# Legacy geometric predicates (maintained for compatibility)
def has_four_straight_lines(f):
    return f.get("num_straight", 0) == 4 or is_square_shape(f)

def exists_quadrangle(f):
    return f.get("has_quadrangle", False) or is_square_shape(f)

def has_obtuse_angle(f):
    return f.get("has_obtuse", False)

# Topology predicates for the 5 shape types
def is_closed_shape(f):
    """Check if shape forms a closed loop"""
    topology = f.get("topology", "")
    return topology in ["closed_loop", "closed_polygon"] or f.get("is_closed", False)

def is_open_shape(f):
    """Check if shape is open (not closed)"""
    topology = f.get("topology", "")
    return topology in ["line_segment", "multi_segment", "open_arc"] and not f.get("is_closed", False)

# Shape complexity predicates
def has_high_complexity(f):
    """Check if shape has high geometric complexity"""
    complexity = f.get("geometric_complexity", 0)
    return complexity > 5 or is_zigzag_shape(f)

def has_low_complexity(f):
    """Check if shape has low geometric complexity"""
    complexity = f.get("geometric_complexity", 0)
    return complexity <= 3 or is_normal_shape(f)

# Symmetry predicates for shape types
def has_rotational_symmetry(f):
    """Check if shape has rotational symmetry"""
    return f.get("rotational_symmetry", False) or f.get("shape_type", "") in ["circle", "square"]

def has_reflection_symmetry(f):
    """Check if shape has reflection symmetry"""
    return f.get("reflection_symmetry", False) or f.get("shape_type", "") in ["square", "triangle"]

# Shape type mapping with enhanced concept predicates
CONCEPTS = {
    # Shape type concepts based on 5 discovered types
    "shape_type_normal": is_normal_shape,
    "shape_type_circle": is_circle_shape,
    "shape_type_square": is_square_shape,
    "shape_type_triangle": is_triangle_shape,
    "shape_type_zigzag": is_zigzag_shape,
    
    # Geometric property concepts
    "has_regular_geometry": has_regular_geometry,
    "has_irregular_geometry": has_irregular_geometry,
    "has_curved_geometry": has_curved_geometry,
    "has_straight_geometry": has_straight_geometry,
    
    # Topology concepts
    "is_closed_shape": is_closed_shape,
    "is_open_shape": is_open_shape,
    
    # Complexity concepts
    "has_high_complexity": has_high_complexity,
    "has_low_complexity": has_low_complexity,
    
    # Symmetry concepts
    "has_rotational_symmetry": has_rotational_symmetry,
    "has_reflection_symmetry": has_reflection_symmetry,
    
    # Legacy concepts (maintained for compatibility)
    "hd_convex-has_four_straight_lines_0010": has_four_straight_lines,
    "hd_exist_quadrangle_0011": exists_quadrangle,
    "hd_closed_shape-has_obtuse_angle_0001": has_obtuse_angle,
    
    # Additional shape-specific concepts
    "has_four_straight_lines": has_four_straight_lines,
    "exists_quadrangle": exists_quadrangle,
    "has_obtuse_angle": has_obtuse_angle,
}
