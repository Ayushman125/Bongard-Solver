# --- Image Preprocessing for Robust Contour Extraction ---
def extract_clean_contours(mask, min_area=10, simplify_epsilon=2.0):
    """
    Given a binary mask (numpy array), extract clean, closed polygons (vertices) using:
    - Contour detection & hierarchy analysis
    - Hole-punching (internal contours)
    - Aggressive thinning & skeletonization
    - Morphological cleaning
    - Contour simplification & filtering
    Returns: List of polygons (each is a list of [x, y] points)
    """
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize, thin
    from skimage.measure import label
    from skimage.morphology import binary_opening, binary_closing
    from skimage.util import img_as_ubyte
    # Step 1: Ensure mask is binary uint8
    mask_bin = (mask > 0).astype(np.uint8)
    # Step 2: Morphological cleaning
    mask_clean = binary_opening(mask_bin, np.ones((3,3)))
    mask_clean = binary_closing(mask_clean, np.ones((3,3)))
    mask_clean = img_as_ubyte(mask_clean)
    # Step 3: Aggressive thinning
    mask_thin = thin(mask_clean)
    mask_thin = img_as_ubyte(mask_thin)
    # Step 4: Skeletonization
    mask_skel = skeletonize(mask_thin > 0)
    mask_skel = img_as_ubyte(mask_skel)
    # Step 5: Contour detection with hierarchy
    contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            # Only use outer contours (hierarchy[i][3] == -1)
            if cv2.contourArea(cnt) >= min_area and hierarchy[i][3] == -1:
                # Step 6: Simplify contour
                epsilon = simplify_epsilon
                cnt_simp = cv2.approxPolyDP(cnt, epsilon, True)
                poly = cnt_simp.reshape(-1, 2).tolist()
                polygons.append(poly)
    # Step 7: Hole-punching (subtract internal contours)
    # (Optional: can be handled by hierarchy, but for complex masks, re-analyze holes)
    # Step 8: Filter out degenerate polygons
    polygons = [poly for poly in polygons if len(poly) >= 3]
    return polygons
from shapely.geometry import Polygon
from typing import Dict, Any
import numpy as np
import logging
import math
from collections import Counter
# TODO: Update these imports with the correct module paths if needed
try:
    from .real_feature_extractor import RealFeatureExtractor, TORCH_KORNIA_AVAILABLE
except ImportError:
    RealFeatureExtractor = None
    TORCH_KORNIA_AVAILABLE = False
try:
    import torch
except ImportError:
    torch = None
# --- Physics Attribute Computation ---
def compute_basic_features(poly: Polygon) -> Dict[str, Any]:
    """Computes basic geometric features for a shapely Polygon."""
    if not poly or not poly.is_valid:
        return {'area': 0.0, 'perimeter': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'orientation': 0.0}

    area = poly.area
    perimeter = poly.length
    centroid = poly.centroid
    minx, miny, maxx, maxy = poly.bounds
    width, height = maxx - minx, maxy - miny
    aspect_ratio = width / height if height > 0 else 1.0

    # Orientation via PCA of vertices
    orientation = 0.0
    try:
        if len(poly.exterior.coords) >= 2:
            coords = np.array(poly.exterior.coords)
            if coords.shape[0] > 1 and coords.shape[1] == 2:
                # Ensure at least 2 points for PCA
                if coords.shape[0] > 1:
                    cov_matrix = np.cov(coords.T)
                    if cov_matrix.shape == (2, 2): # Ensure it's a 2x2 matrix
                        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                        # The first principal component (eigenvector corresponding to largest eigenvalue)
                        # gives the direction of the major axis.
                        major_axis_idx = np.argmax(eigenvalues)
                        principal_axis = eigenvectors[:, major_axis_idx]
                        orientation = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))) % 360
    except Exception as e:
        logging.debug(f"Could not compute orientation for polygon: {e}")
        orientation = 0.0 # Default to 0 on error

    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'cx': float(centroid.x),
        'cy': float(centroid.y),
        'bbox': [float(minx), float(miny), float(maxx), float(maxy)],
        'aspect_ratio': float(aspect_ratio),
        'orientation': float(orientation)
    }

def compute_physics_attributes(node_data):
    """
    Computes robust physics attributes and asserts their domain validity.
    Now also computes perimeter and compactness.
    """
    if not isinstance(node_data, dict):
        logging.error(f"compute_physics_attributes: node_data is not a dict: type={type(node_data)}, value={repr(node_data)}")
        # Initialize with default values to prevent further errors
        node_data.update({'area': 0.0, 'inertia': 0.0, 'convexity': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'perimeter': 0.0, 'orientation': 0.0, 'compactness': 0.0, 'num_segments': 0, 'num_junctions': 0})
        return

    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} FULL INPUT: {node_data}")
    vertices = node_data.get('vertices')
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} input vertices: {vertices}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} input centroid: {node_data.get('centroid', None)}")

    # --- Advanced Polygon Repair and Validation ---
    valid_poly = False
    repaired = False
    repaired_reason = None
    orig_vertices = vertices
    poly = None
    import warnings
    # Remove duplicate points
    if vertices:
        unique_vertices = []
        seen = set()
        for pt in vertices:
            tpt = tuple(np.round(pt, 6)) if isinstance(pt, (list, tuple, np.ndarray)) else tuple(pt)
            if tpt not in seen:
                unique_vertices.append(pt)
                seen.add(tpt)
        vertices = unique_vertices
        arr = np.array(vertices)
        # Try direct polygon
        if len(vertices) >= 3:
            try:
                poly = Polygon(vertices)
                if poly.is_valid and poly.area > 0:
                    valid_poly = True
            except Exception as e:
                warnings.warn(f"Polygon creation failed: {e}")
        # Self-intersection repair
        if not valid_poly and len(vertices) >= 3:
            try:
                poly = Polygon(vertices).buffer(0)
                if poly.is_valid and poly.area > 0:
                    valid_poly = True
                    repaired = True
                    repaired_reason = 'self-intersection repair (buffer(0))'
            except Exception as e:
                warnings.warn(f"Self-intersection repair failed: {e}")
        # Convex hull
        if not valid_poly and len(vertices) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(arr)
                poly = Polygon(arr[hull.vertices])
                if poly.is_valid and poly.area > 0:
                    valid_poly = True
                    repaired = True
                    repaired_reason = 'convex hull'
            except Exception as e:
                warnings.warn(f"Convex hull repair failed: {e}")
        # Alpha shape (concave hull)
        if not valid_poly and len(vertices) >= 4:
            try:
                import alphashape
                alpha_poly = alphashape.alphashape(arr, 0.1)
                if alpha_poly.is_valid and alpha_poly.area > 0:
                    poly = alpha_poly
                    valid_poly = True
                    repaired = True
                    repaired_reason = 'alpha shape'
            except Exception as e:
                warnings.warn(f"Alpha shape repair failed: {e}")
        # Collinearity detection: fallback to min-area rectangle
        if not valid_poly and len(vertices) >= 3:
            v0, v1, v2 = arr[0], arr[1], arr[2]
            area_test = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0))
            if area_test < 1e-6:
                minx, miny = arr.min(axis=0)
                maxx, maxy = arr.max(axis=0)
                poly = Polygon([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
                valid_poly = True
                repaired = True
                repaired_reason = 'collinear points, min-area rectangle'
        # Skeleton-based reconstruction (if mask available)
        if not valid_poly and 'mask' in node_data:
            try:
                polygons = extract_clean_contours(node_data['mask'])
                if polygons:
                    poly = Polygon(polygons[0])
                    if poly.is_valid and poly.area > 0:
                        valid_poly = True
                        repaired = True
                        repaired_reason = 'mask-based contour extraction'
            except Exception as e:
                warnings.warn(f"Mask-based contour extraction failed: {e}")
        # Multi-scale contour extraction (if mask available)
        if not valid_poly and 'mask' in node_data:
            try:
                scales = [1.0, 0.5, 1.5]
                polygons = []
                import cv2
                mask = node_data['mask']
                for scale in scales:
                    mask_scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    polys = extract_clean_contours(mask_scaled)
                    polygons.extend(polys)
                polygons = [poly for poly in polygons if len(poly) >= 3]
                if polygons:
                    poly = Polygon(polygons[0])
                    if poly.is_valid and poly.area > 0:
                        valid_poly = True
                        repaired = True
                        repaired_reason = 'multi-scale contour extraction'
            except Exception as e:
                warnings.warn(f"Multi-scale contour extraction failed: {e}")
    # If polygon is still invalid, do not assign attributes and log a warning
    if not valid_poly:
        logging.warning(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} has invalid geometry and cannot be repaired. No attributes assigned.")
        return
    # Now compute attributes as usual
    basic_features = compute_basic_features(poly)
    node_data.update(basic_features)
    area = node_data['area']
    assert area >= 0.0, f"Negative area {area} for node {node_data.get('id')}"
    try:
        inertia = poly.moment_of_inertia
    except Exception:
        inertia = 0.0
    convexity = poly.convex_hull.area / area if area > 0 else 0.0
    perimeter = node_data['perimeter']
    compactness = perimeter**2 / (4 * math.pi * area) if area > 0 else 0.0
    action_program = node_data.get('action_program', [])
    num_segments = len(extract_line_segments(action_program))
    # Log all computed attributes for diagnostics
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed area: {area}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed inertia: {inertia}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed convexity: {convexity}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed perimeter: {perimeter}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed compactness: {compactness}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed num_segments: {num_segments}")
    # Handle Polygon and MultiPolygon for junction counting and feature aggregation
    num_junctions = 0
    if hasattr(poly, 'exterior'):
        # Polygon case
        coords_counter = Counter(tuple(c) for c in poly.exterior.coords)
        num_junctions = sum(1 for count in coords_counter.values() if count > 1)
        node_data.update({
            'inertia': float(inertia),
            'convexity': float(convexity),
            'compactness': float(compactness),
            'num_segments': int(num_segments),
            'num_junctions': int(num_junctions),
        })
        logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} FULL OUTPUT: {node_data}")
    elif getattr(poly, 'geom_type', None) == 'MultiPolygon':
        # MultiPolygon: aggregate features from all sub-polygons
        total_area = 0.0
        total_perimeter = 0.0
        total_inertia = 0.0
        total_convexity = 0.0
        total_compactness = 0.0
        total_num_junctions = 0
        for subpoly in poly.geoms:
            sub_features = compute_basic_features(subpoly)
            sub_area = sub_features.get('area', 0.0)
            sub_perimeter = sub_features.get('perimeter', 0.0)
            try:
                sub_inertia = subpoly.moment_of_inertia
            except Exception:
                sub_inertia = 0.0
            sub_convexity = subpoly.convex_hull.area / sub_area if sub_area > 0 else 0.0
            sub_compactness = sub_perimeter**2 / (4 * math.pi * sub_area) if sub_area > 0 else 0.0
            sub_num_junctions = 0
            if hasattr(subpoly, 'exterior'):
                coords_counter = Counter(tuple(c) for c in subpoly.exterior.coords)
                sub_num_junctions = sum(1 for count in coords_counter.values() if count > 1)
            total_area += sub_area
            total_perimeter += sub_perimeter
            total_inertia += sub_inertia
            total_convexity += sub_convexity
            total_compactness += sub_compactness
            total_num_junctions += sub_num_junctions
        node_data.update({
            'area': float(total_area),
            'perimeter': float(total_perimeter),
            'inertia': float(total_inertia),
            'convexity': float(total_convexity),
            'compactness': float(total_compactness),
            'num_segments': int(num_segments),
            'num_junctions': int(total_num_junctions),
        })
        logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} FULL OUTPUT (MultiPolygon): {node_data}")
    else:
        logging.warning(f"compute_physics_attributes: Unknown geometry type {getattr(poly, 'geom_type', None)} for node {node_data.get('id', None)}")

    # Compute basic features first
    basic_features = compute_basic_features(poly)
    node_data.update(basic_features)

    area = node_data['area'] # Use the computed area
    assert area >= 0.0, f"Negative area {area} for node {node_data.get('id')}"

    try:
        inertia = poly.moment_of_inertia
    except Exception:
        inertia = 0.0

    convexity = poly.convex_hull.area / area if area > 0 else 0.0
    perimeter = node_data['perimeter']
    compactness = perimeter**2 / (4 * math.pi * area) if area > 0 else 0.0
    action_program = node_data.get('action_program', [])
    num_segments = len(extract_line_segments(action_program))

    # Log all computed attributes for diagnostics
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed area: {area}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed inertia: {inertia}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed convexity: {convexity}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed perimeter: {perimeter}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed compactness: {compactness}")
    logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} computed num_segments: {num_segments}")
    
    # ...existing code...

def extract_line_segments(action_program):
    """
    Extracts line segments from a simplified action program.
    Assumes action_program is a list of commands like:
    [{'type': 'start', 'x': x1, 'y': y1}, {'type': 'line', 'x': x2, 'y': y2}, ...]
    """
    segments = []
    current_point = None
    if not isinstance(action_program, list):
        return [] # Return empty if not a list

    for command in action_program:
        if command['type'] == 'start':
            current_point = (command['x'], command['y'])
        elif command['type'] == 'line' and current_point:
            next_point = (command['x'], command['y'])
            segments.append((current_point, next_point))
            current_point = next_point
        # Add other command types if necessary (e.g., 'arc', 'curve')
    return segments

# --- Singleton for RealFeatureExtractor ---
_REAL_FEATURE_EXTRACTOR_INSTANCE = None
def get_real_feature_extractor():
    global _REAL_FEATURE_EXTRACTOR_INSTANCE
    if _REAL_FEATURE_EXTRACTOR_INSTANCE is not None:
        return _REAL_FEATURE_EXTRACTOR_INSTANCE
    if not TORCH_KORNIA_AVAILABLE:
        logging.warning("Torch/Kornia/Torchvision not available. RealFeatureExtractor will not be initialized.")
        return None
    try:
        _REAL_FEATURE_EXTRACTOR_INSTANCE = RealFeatureExtractor(
            clip_model_name="openai/clip-vit-base-patch32",
            sam_encoder_path="sam_checkpoints/sam_vit_h_4b8939.pth",
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_features=True
        )
        logging.info("RealFeatureExtractor singleton initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize RealFeatureExtractor: {e}. Feature extraction will be skipped.")
        _REAL_FEATURE_EXTRACTOR_INSTANCE = None
    return _REAL_FEATURE_EXTRACTOR_INSTANCE
