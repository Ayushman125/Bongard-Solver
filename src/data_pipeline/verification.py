import logging
import numpy as np

def has_quadrangle(poly_geom):
        # True if convex hull has 4 sides
        from src.data_pipeline.physics_infer import PhysicsInference
        try:
            poly = PhysicsInference._ensure_polygon(poly_geom)
            hull = poly.convex_hull
            coords = list(hull.exterior.coords)
            return len(coords) - 1 == 4
        except Exception:
            return False

def has_obtuse_angle(poly_geom):
        # True if any interior angle > 90°
        from src.data_pipeline.physics_infer import PhysicsInference
        import math
        poly = PhysicsInference._ensure_polygon(poly_geom)
        coords = list(poly.exterior.coords)
        n = len(coords) - 1
        for i in range(n):
            a, b, c = coords[i-1], coords[i], coords[(i+1)%n]
            ab = (b[0]-a[0], b[1]-a[1])
            cb = (b[0]-c[0], b[1]-c[1])
            dot = ab[0]*cb[0] + ab[1]*cb[1]
            norm = math.hypot(*ab) * math.hypot(*cb)
            if norm == 0:
                continue
            angle = math.degrees(math.acos(dot/norm))
            if angle > 90:
                return True
        return False

def validate_polygon(poly):
        if not poly.is_valid:
            logging.warning(f"Invalid polygon detected: {poly.wkt}")
            return False
        return True

def log_for_review(problem_id, filename, reason, review_log='data/flagged_cases.txt'):
        with open(review_log, 'a') as f:
            f.write(f"{problem_id},{filename},{reason}\n")
    
def has_min_vertices(vertices, min_v=4):
    # ensure polygon closure & ≥ min_v distinct points
    if not isinstance(vertices, list):
        return False
    unique = [v for i, v in enumerate(vertices) if i == 0 or v != vertices[i-1]]
    return len(unique) >= min_v

def l2_shape_distance(vtx_a, vtx_b):
    # quick Hausdorff-like proxy
    if not (isinstance(vtx_a, list) and isinstance(vtx_b, list)):
        return float('inf')
    if len(vtx_a) == 0 or len(vtx_b) == 0:
        return float('inf')
    pa = np.array(vtx_a[:20])
    pb = np.array(vtx_b[:20])
    if pa.shape != pb.shape:
        # Pad the shorter one
        if pa.shape[0] < pb.shape[0]:
            pa = np.pad(pa, ((0, pb.shape[0] - pa.shape[0]), (0, 0)), mode='edge')
        else:
            pb = np.pad(pb, ((0, pa.shape[0] - pb.shape[0]), (0, 0)), mode='edge')
    return np.mean(np.linalg.norm(pa - pb, axis=1))
