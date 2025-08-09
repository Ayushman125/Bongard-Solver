import logging
from typing import Dict, List, Any, Optional
from src.physics_inference import PhysicsInference

def standardize_coordinates(vertices, target_range=(0, 1)):
    """Ensure all vertices are in consistent coordinate system [target_range]."""
    import numpy as np
    import logging
    logging.info(f"[standardize_coordinates] INPUT vertices: {vertices}")
    if not vertices or len(vertices) < 2:
        logging.warning(f"[standardize_coordinates] Not enough vertices to standardize: {vertices}")
        return vertices
    arr = np.array(vertices)
    min_x, min_y = np.min(arr, axis=0)
    max_x, max_y = np.max(arr, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    if width > 0 and height > 0:
        normalized = (arr - [min_x, min_y]) / [width, height]
        logging.info(f"[standardize_coordinates] OUTPUT normalized vertices: {normalized.tolist()}")
        return normalized.tolist()
    logging.warning(f"[standardize_coordinates] Degenerate width/height, returning arr: {arr.tolist()}")
    return arr.tolist()

def calculate_geometry_consistent(vertices):
    """Calculate geometry with consistent coordinate normalization."""
    from shapely.geometry import Polygon
    import logging
    logging.info(f"[calculate_geometry_consistent] INPUT vertices: {vertices}")
    if not vertices or len(vertices) < 3:
        logging.warning(f"[calculate_geometry_consistent] Not enough vertices for geometry: {vertices}")
        # PATCH: Always include width and height in output dict
        return {'area': 0, 'perimeter': 0, 'centroid': [0, 0], 'bounds': [0, 0, 0, 0], 'width': 0.0, 'height': 0.0}
    try:
        poly = Polygon(vertices)
        min_x, min_y, max_x, max_y = poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        geometry = {
            'area': poly.area,
            'perimeter': poly.length,
            'centroid': list(poly.centroid.coords[0]),
            'bounds': list(poly.bounds),
            'width': width,
            'height': height
        }
        logging.info(f"[calculate_geometry_consistent] OUTPUT geometry: {geometry}")
        return geometry
    except Exception as e:
        logging.error(f"[calculate_geometry_consistent] Exception: {e}")
        # PATCH: Always include width and height in output dict
        return {'area': 0, 'perimeter': 0, 'centroid': [0, 0], 'bounds': [0, 0, 0, 0], 'width': 0.0, 'height': 0.0}

def calculate_complexity(vertices: List[tuple]) -> float:
    """
    Compute a standardized shape complexity metric as a function of:
      - Number of vertices (normalized)
      - Curvature score (normalized)
      - Irregularity (normalized)
      - Optionally, compactness (inverse)
    All features are computed from the normalized, deduplicated, and complete set of vertices.
    Returns a value in [0, 1], higher = more complex.
    """
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)
    try:
        verts = normalize_vertices(vertices)
        n = len(verts)
        if n < 3:
            logger.info(f"Complexity: <3 vertices, returning 0.0")
            return 0.0
        # Number of vertices (normalized: 0 for triangle, 1 for >=20)
        n_norm = min(1.0, (n - 3) / 17.0)
        # Curvature score (normalized: 0 = line, 1 = highly curved)
        curvature = _calculate_curvature_score(verts)
        curvature_norm = min(1.0, curvature / 2.0)  # Empirical scaling
        # Irregularity (already normalized 0-1)
        irregularity = _calculate_irregularity(verts)
        # Optionally, compactness (inverse: 0 = circle, 1 = non-compact)
        geom = calculate_geometry(verts)
        compactness = _calculate_compactness(geom.get('area', 0.0), geom.get('perimeter', 0.0))
        compactness_inv = 1.0 - min(1.0, compactness) if compactness == compactness else 1.0  # NaN-safe
        # Weighted sum (weights can be tuned)
        complexity = 0.3 * n_norm + 0.3 * curvature_norm + 0.2 * irregularity + 0.2 * compactness_inv
        logger.info(f"Complexity: n={n}, n_norm={n_norm:.2f}, curvature={curvature:.2f}, curvature_norm={curvature_norm:.2f}, irregularity={irregularity:.2f}, compactness_inv={compactness_inv:.2f}, result={complexity:.2f}")
        return float(np.clip(complexity, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"calculate_complexity failed: {e}")
        return 0.0


def _calculate_compactness(area: float, perimeter: float) -> float:
    """
    Isoperimetric ratio: (4πA)/P². Returns 1 for a perfect circle, <1 otherwise.
    No clamping or bounding is applied. Returns NaN for degenerate cases.
    """
    import math
    try:
        if area is None or perimeter is None or perimeter == 0 or area <= 0:
            return float('nan')
        return (4 * math.pi * area) / (perimeter ** 2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Compactness error: {e}")
        return float('nan')

def _calculate_angular_variance(vertices: list) -> float:
    # Use robust angular variance, fallback to 0 for <3 points
    return PhysicsInference.robust_angular_variance(vertices)


def _calculate_pattern_regularity_from_modifiers(modifier_sequence: list) -> float:
    """Pattern regularity using PhysicsInference.pattern_regularity. Returns NaN if sequence too short."""
    return PhysicsInference.pattern_regularity(modifier_sequence)
def _check_horizontal_symmetry(vertices, poly=None):
    """Check horizontal symmetry using PhysicsInference or geometric comparison."""
    try:
        # Prefer PhysicsInference if available
        if hasattr(PhysicsInference, 'horizontal_symmetry'):
            return PhysicsInference.horizontal_symmetry(vertices)
        # Fallback: compare top and bottom halves
        if poly is not None and hasattr(poly, 'centroid'):
            centroid_y = poly.centroid.y
        else:
            centroid_y = sum(v[1] for v in vertices) / len(vertices)
        reflected = [(x, 2*centroid_y - y) for x, y in vertices]
        # Compare original and reflected (simple mean distance)
        import numpy as np
        orig = np.array(vertices)
        refl = np.array(reflected)
        if orig.shape == refl.shape:
            return float(np.mean(np.linalg.norm(orig - refl, axis=1)))
        return 0.0
    except Exception as e:
        logging.getLogger(__name__).warning(f"Horizontal symmetry error: {e}")
        return 0.0

def _check_vertical_symmetry(vertices, poly=None):
    """Check vertical symmetry using PhysicsInference or geometric comparison."""
    try:
        if hasattr(PhysicsInference, 'vertical_symmetry'):
            return PhysicsInference.vertical_symmetry(vertices)
        if poly is not None and hasattr(poly, 'centroid'):
            centroid_x = poly.centroid.x
        else:
            centroid_x = sum(v[0] for v in vertices) / len(vertices)
        reflected = [(2*centroid_x - x, y) for x, y in vertices]
        import numpy as np
        orig = np.array(vertices)
        refl = np.array(reflected)
        if orig.shape == refl.shape:
            return float(np.mean(np.linalg.norm(orig - refl, axis=1)))
        return 0.0
    except Exception as e:
        logging.getLogger(__name__).warning(f"Vertical symmetry error: {e}")
        return 0.0

def _calculate_edge_length_variance(vertices):
    """Population variance of edge lengths, see PhysicsInference.edge_length_variance."""
    return PhysicsInference.edge_length_variance(vertices)

def normalize_vertices(vertices_raw):
    """
    Normalize coordinates to [0,1] in both axes, preserving aspect ratio and centering shape if needed.
    Uses PhysicsInference.dedup_vertices and PhysicsInference.rounded_bbox.
    """
    import numpy as np
    import logging
    logging.info(f"[normalize_vertices] INPUT vertices_raw: {vertices_raw}")
    verts = ensure_vertex_list(vertices_raw)
    verts = PhysicsInference.dedup_vertices(verts)
    if len(verts) < 2:
        logging.warning(f"[normalize_vertices] Not enough vertices to normalize: {verts}")
        return verts
    minx, miny, maxx, maxy = PhysicsInference.rounded_bbox(verts)
    width = maxx - minx
    height = maxy - miny
    if width < 1e-8 or height < 1e-8:
        logging.warning(f"[normalize_vertices] Degenerate width/height, returning verts: {verts}")
        return verts
    arr = np.array(verts)
    arr = (arr - [minx, miny]) / [width, height]
    arr[np.abs(arr) < 1e-10] = 0.0
    normalized = [tuple(pt) for pt in arr]
    logging.info(f"[normalize_vertices] OUTPUT normalized vertices: {normalized}")
    return normalized

def calculate_geometry(vertices):
    """Calculate geometry properties from normalized vertices, robustly constructing polygon."""
    import numpy as np
    import logging
    logging.info(f"[calculate_geometry] INPUT vertices: {vertices}")
    # PATCH: Validate and correct input vertices
    if not vertices or len(vertices) < 3:
        logging.warning(f"[calculate_geometry] PATCH: Not enough vertices for geometry: {vertices}")
        return {
            'bbox': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
            'centroid': [0.0, 0.0],
            'width': 0.0,
            'height': 0.0,
            'area': 0.0,
            'perimeter': 0.0,
            'moment_of_inertia': 0.0,
            'convexity_ratio': 0.0
        }
    # Remove duplicate consecutive points
    deduped = [vertices[0]]
    for pt in vertices[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    verts = deduped
    # Close polygon if not closed
    if verts[0] != verts[-1]:
        verts.append(verts[0])
    logging.info(f"[calculate_geometry] PATCH: Validated/corrected vertices: {verts}")
    verts = normalize_vertices(list(verts))
    if len(verts) < 3:
        logging.warning(f"[calculate_geometry] PATCH: Normalization collapsed points: {verts}")
        return {
            'bbox': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
            'centroid': [0.0, 0.0],
            'width': 0.0,
            'height': 0.0,
            'area': 0.0,
            'perimeter': 0.0,
            'moment_of_inertia': 0.0,
            'convexity_ratio': 0.0
        }
    xs, ys = zip(*verts)
    bbox = {'min_x': min(xs), 'max_x': max(xs), 'min_y': min(ys), 'max_y': max(ys)}
    width = bbox['max_x'] - bbox['min_x']
    height = bbox['max_y'] - bbox['min_y']
    area = PhysicsInference.shoelace_area(verts) if len(verts) >= 3 else 0.0
    try:
        from shapely.geometry import Polygon
        perimeter = 0.0
        poly = None
        if len(verts) >= 3:
            poly = Polygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            perimeter = poly.length
            centroid = list(poly.centroid.coords[0])
            area_val = poly.area
        elif len(verts) == 2:
            perimeter = float(np.linalg.norm(np.array(verts[1]) - np.array(verts[0])))
            centroid = PhysicsInference.centroid(verts)
            area_val = area
        else:
            centroid = PhysicsInference.centroid(verts) if len(verts) >= 1 else [0.0, 0.0]
            area_val = area
        inertia = PhysicsInference.moment_of_inertia(verts) if len(verts) >= 2 else 0.0
        # Robust convexity: only defined for >=3 points, else 0.0
        if len(verts) >= 3:
            try:
                convexity = PhysicsInference.convexity_ratio(verts)
                if convexity != convexity:  # NaN check
                    convexity = 0.0
            except Exception:
                convexity = 0.0
        else:
            convexity = 0.0
        # Ensure all outputs are JSON-safe
        def _json_safe(x):
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                return 0.0
            return float(x) if isinstance(x, float) else x
        geometry = {
            'bbox': {k: _json_safe(v) for k, v in bbox.items()},
            'centroid': [_json_safe(c) for c in centroid],
            'width': _json_safe(width),
            'height': _json_safe(height),
            'area': _json_safe(area_val),
            'perimeter': _json_safe(perimeter),
            'moment_of_inertia': _json_safe(inertia),
            'convexity_ratio': _json_safe(convexity)
        }
        logging.info(f"[calculate_geometry] OUTPUT geometry: {geometry}")
        return geometry
    except Exception as e:
        logging.error(f"[calculate_geometry] Exception: {e}")
        geometry = {
            'bbox': bbox,
            'centroid': [0.0, 0.0],
            'width': width,
            'height': height,
            'area': area,
            'perimeter': 0.0,
            'moment_of_inertia': 0.0,
            'convexity_ratio': 0.0
        }
        return geometry

def extract_position_and_rotation(vertices):
    """Given a list of (x, y) normalized vertices, return centroid and orientation angle (degrees)."""
    import numpy as np
    try:
        pts = np.array(vertices)
        if pts.shape[0] < 2:
            return {'centroid': [float(pts[0,0]), float(pts[0,1])] if pts.shape[0] == 1 else [0.5, 0.5], 'orientation_degrees': 0.0}
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        # PCA: first principal axis
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        angle = np.degrees(np.arctan2(axis[1], axis[0]))
        return {
            'centroid': [float(centroid[0]), float(centroid[1])],
            'orientation_degrees': float(angle)
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"extract_position_and_rotation failed: {e}")
        return {'centroid': [0.5, 0.5], 'orientation_degrees': 0.0}

def ensure_vertex_list(vertices):
    """Convert Polygon or similar geometry object to list of tuples."""
    if hasattr(vertices, 'exterior') and hasattr(vertices.exterior, 'coords'):
        return list(vertices.exterior.coords)
    elif hasattr(vertices, 'coords'):
        return list(vertices.coords)
    return vertices
    
def safe_divide(a, b, default=0.0):
    """Safe division avoiding zero/NaN."""
    if abs(b) < 1e-10:
        return default
    return a / b

def json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    # If it's a custom object (e.g., LineAction, ArcAction), convert to robust string
    elif hasattr(obj, '__str__') and not isinstance(obj, (str, bytes)):
        logger.debug(f"[json_safe] Converting custom object to string: {type(obj).__name__}")
        return str(obj)
    return obj

def _calculate_homogeneity_score(modifier_sequence: list) -> float:
    """Homogeneity score using PhysicsInference.homogeneity_score (Simpson's index)."""
    return PhysicsInference.homogeneity_score(modifier_sequence)

def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values. For length 2, return squared diff. For length 1, return NaN and log."""
    import numpy as np
    logger = logging.getLogger(__name__)
    n = len(values)
    if n < 1:
        logger.warning("Variance: empty list, returning NaN")
        return float('nan')
    if n == 1:
        logger.warning("Variance: only one value, returning NaN")
        return float('nan')
    if n == 2:
        diff = values[1] - values[0]
        return diff * diff / 2.0
    mean = safe_divide(sum(values), n)
    return safe_divide(sum((x - mean) ** 2 for x in values), n)
    
def _calculate_dominant_direction(line_features: List[Dict]) -> str:
    """Calculate the dominant direction of line strokes"""
    if not line_features:
        return 'none'
    directions = [f['line_direction'] for f in line_features]
    direction_counts = {}
    for direction in directions:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    return max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else 'none'


def _calculate_perimeter(vertices: List[tuple]) -> float:
    """Calculate perimeter of the shape."""
    # Fix: Ensure vertices is a list of tuples, not a Polygon object
    vertices = ensure_vertex_list(vertices)
    if len(vertices) < 2:
        return 0.0
    perimeter = 0.0
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        perimeter += ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
    return json_safe(perimeter)

def _calculate_convexity_ratio(poly) -> float:
    """Calculate ratio of polygon area to convex hull area."""
    try:
        if poly.area == 0:
            return 0.0
        return safe_divide(poly.area, poly.convex_hull.area)
    except:
        return 0.0

# (Removed duplicate _calculate_compactness)

def _calculate_eccentricity(vertices: List[tuple]) -> float:
    """Calculate eccentricity as 1 - (min_eigenvalue / max_eigenvalue) from PCA."""
    import numpy as np
    vertices = ensure_vertex_list(vertices)
    if len(vertices) < 3:
        return 0.0
    try:
        points = np.array(vertices)
        centered = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(np.abs(eigenvals))[::-1]
        if eigenvals[0] == 0:
            return 0.0
        return json_safe(1.0 - safe_divide(eigenvals[-1], eigenvals[0]))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Eccentricity error: {e}")
        return 0.0


def _check_rotational_symmetry(vertices: List[tuple]) -> int:
    """Check rotational symmetry order using k-fold RMSE (k=2,3,4)."""
    return PhysicsInference.rotational_symmetry(vertices)

def _calculate_irregularity(vertices: List[tuple]) -> float:
    """Calculate normalized mean absolute deviation from regular n-gon angle (0=regular, 1=irregular)."""
    import numpy as np
    vertices = ensure_vertex_list(vertices)
    n = len(vertices)
    if n < 3:
        return 0.0
    try:
        angles = []
        for i in range(n):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i + 1) % n])
            p3 = np.array(vertices[(i + 2) % n])
            v1 = p2 - p1
            v2 = p3 - p2
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angles.append(angle)
        if not angles:
            return 0.0
        expected_angle = safe_divide((n - 2) * np.pi, n)
        mad = np.mean([abs(angle - expected_angle) for angle in angles])
        # Normalize: 0 = regular, 1 = max deviation (pi)
        norm_mad = min(1.0, mad / np.pi)
        return json_safe(norm_mad)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Irregularity error: {e}")
        return 0.0
def _calculate_curvature_score(vertices: list) -> float:
    """Curvature score: average absolute change in tangent angle per unit length (see PhysicsInference.curvature_score)."""
    return PhysicsInference.curvature_score(vertices)

def _calculate_homogeneity(modifier_distribution: dict) -> float:
    """Calculate a simple homogeneity score: 1.0 if all modifiers are the same, lower otherwise (Gini impurity)."""
    total = sum(modifier_distribution.values())
    if total == 0:
        return 1.0
    probs = [v / total for v in modifier_distribution.values()]
    gini = 1.0 - sum(p ** 2 for p in probs)
    return 1.0 - gini  # 1.0 = homogeneous, 0 = maximally diverse

def ensure_flat_str_list(obj):
    """Recursively flattens any nested list/tuple and ensures all items are str."""
    out = []
    if isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(ensure_flat_str_list(x))
    elif hasattr(obj, 'raw_command') and isinstance(obj.raw_command, str):
        out.append(obj.raw_command)
    else:
        out.append(str(obj))
    return out

