# Problem-aware concept predicates for Bongard-Solver
# Add more rules as needed for your problem set

def has_four_straight_lines(f):
    return f.get("num_straight", 0) == 4

def exists_quadrangle(f):
    return f.get("has_quadrangle", False)

def has_obtuse_angle(f):
    return f.get("has_obtuse", False)

# Example mapping; extend for all problems
CONCEPTS = {
    "hd_convex-has_four_straight_lines_0010": has_four_straight_lines,
    "hd_exist_quadrangle_0011": exists_quadrangle,
    "hd_closed_shape-has_obtuse_angle_0001": has_obtuse_angle,
    # ... add more mappings ...
}
