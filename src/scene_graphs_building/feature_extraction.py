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

# --- Geometry validity logic update ---
def set_geometry_valid(obj):
    # Broaden geometry_valid: allow polygons, lines, arcs, and points as valid geometry
    if obj.get('object_type') in ('polygon', 'line', 'arc', 'point'):
        obj['geometry_valid'] = True
    else:
        obj['geometry_valid'] = False
    return obj
from shapely.geometry import Polygon
from typing import Dict, Any
import numpy as np
import logging
import math
from collections import Counter
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
    # --- Simplified, robust LOGO/NVLabs geometry handling ---
    vertices = node_data.get('vertices', [])
    object_type = node_data.get('object_type', None)
    is_closed = node_data.get('is_closed', False)
    action_program = node_data.get('action_program', [])
    # --- Extract stroke-level programmatic metadata ---
    # If action_program is a list of dicts, extract per-stroke fields
    if isinstance(action_program, list) and all(isinstance(cmd, dict) for cmd in action_program):
        node_data['stroke_type'] = action_program[0].get('stroke_type', None) if action_program else None
        node_data['turn_direction'] = action_program[0].get('turn_direction', None) if action_program else None
        node_data['turn_angle'] = action_program[0].get('turn_angle', None) if action_program else None
        node_data['action_index'] = action_program[0].get('action_index', None) if action_program else None
        node_data['repetition_count'] = action_program[0].get('repetition_count', None) if action_program else None
    else:
        node_data['stroke_type'] = None
        node_data['turn_direction'] = None
        node_data['turn_angle'] = None
        node_data['action_index'] = None
        node_data['repetition_count'] = None
    # --- Curvature type: classify as line or arc ---
    if object_type == 'line':
        node_data['curvature_type'] = 'line'
    elif object_type == 'arc':
        node_data['curvature_type'] = 'arc'
    else:
        node_data['curvature_type'] = None
    # --- Length, orientation, centroid for all strokes (lines, polygons, arcs, etc.) ---
    if vertices and len(vertices) >= 2:
        arr = np.array(vertices)
        node_data['length'] = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
        # For lines/arcs, orientation is from first to last point
        if node_data.get('object_type') in ('line', 'arc'):
            node_data['orientation'] = float(np.degrees(np.arctan2(arr[-1][1] - arr[0][1], arr[-1][0] - arr[0][0])))
            node_data['centroid'] = arr.mean(axis=0).tolist()
        else:
            # Principal direction via PCA for orientation
            try:
                cov_matrix = np.cov(arr.T)
                if cov_matrix.shape == (2, 2):
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    major_axis_idx = np.argmax(eigenvalues)
                    principal_axis = eigenvectors[:, major_axis_idx]
                    orientation = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))) % 360
                    node_data['orientation'] = orientation
            except Exception as e:
                node_data['orientation'] = None
    else:
        node_data['length'] = None
        node_data['orientation'] = None
    # Geometry valid if polygon and closed
    geometry_valid = object_type == 'polygon' and is_closed and vertices and len(vertices) >= 3
    node_data['geometry_valid'] = geometry_valid
    # Per-feature valid flags
    feature_valid = {}
    # List of all required features and their valid flags
    required_features = [
        'centroid', 'cx', 'cy', 'area', 'perimeter', 'bbox', 'aspect_ratio', 'orientation',
        'compactness', 'inertia', 'convexity', 'num_segments', 'num_junctions', 'curvature',
        'skeleton_length', 'symmetry_axis', 'length', 'curvature_type', 'stroke_type',
        'turn_direction', 'turn_angle', 'action_index', 'repetition_count'
    ]
    required_valid_flags = [
        'curvature_valid','skeleton_length_valid','symmetry_axis_valid','centroid_valid','area_valid',
        'orientation_valid','aspect_ratio_valid','perimeter_valid','compactness_valid','convexity_valid',
        'inertia_valid','num_segments_valid','num_junctions_valid'
    ]
    # Polygon and closed: compute all features
    if geometry_valid:
        poly = Polygon(vertices)
        if poly.is_valid and poly.area > 0:
            centroid = [float(poly.centroid.x), float(poly.centroid.y)]
            node_data['centroid'] = centroid
            node_data['cx'] = centroid[0]
            node_data['cy'] = centroid[1]
            node_data['fallback_geometry'] = False
            basic_features = compute_basic_features(poly)
            node_data.update(basic_features)
            area = node_data['area']
            try:
                inertia = poly.moment_of_inertia
            except Exception:
                inertia = None
            convexity = poly.convex_hull.area / area if area > 0 else None
            perimeter = node_data['perimeter']
            compactness = perimeter**2 / (4 * math.pi * area) if area > 0 else None
            action_program = node_data.get('action_program', [])
            if (
                isinstance(action_program, list)
                and all(isinstance(cmd, dict) for cmd in action_program)
            ):
                num_segments = len(extract_line_segments(action_program))
            else:
                num_segments = None
            coords_counter = Counter(tuple(c) for c in poly.exterior.coords)
            num_junctions = sum(1 for count in coords_counter.values() if count > 1)
            node_data.update({
                'inertia': inertia,
                'convexity': convexity,
                'compactness': compactness,
                'num_segments': int(num_segments) if num_segments is not None else None,
                'num_junctions': int(num_junctions),
            })
            # Curvature
            arr = np.array(vertices)
            angles = []
            for i in range(len(arr)):
                p_prev = arr[i-1]
                p_curr = arr[i]
                p_next = arr[(i+1)%len(arr)]
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            if angles:
                curvature = float(np.mean(np.abs(angles)))
                node_data['curvature'] = curvature
                feature_valid['curvature_valid'] = True
            else:
                node_data['curvature'] = None
                feature_valid['curvature_valid'] = False
            # Symmetry axis
            symmetry_axis = None
            try:
                if len(vertices) >= 2:
                    arr = np.array(vertices)
                    cov_matrix = np.cov(arr.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    major_axis_idx = np.argmax(eigenvalues)
                    principal_axis = eigenvectors[:, major_axis_idx]
                    centroid = np.mean(arr, axis=0)
                    symmetry_axis = {
                        'centroid': centroid.tolist(),
                        'direction': principal_axis.tolist()
                    }
            except Exception as e:
                logging.warning(f"Symmetry axis computation failed: {e}")
            node_data['symmetry_axis'] = symmetry_axis
            feature_valid['symmetry_axis_valid'] = symmetry_axis is not None
            # Skeleton length: not needed for LOGO data
            node_data['skeleton_length'] = None
            feature_valid['skeleton_length_valid'] = False
            feature_valid['centroid_valid'] = True
            feature_valid['area_valid'] = True
            feature_valid['orientation_valid'] = True
            feature_valid['aspect_ratio_valid'] = True
            feature_valid['perimeter_valid'] = True
            feature_valid['compactness_valid'] = compactness is not None
            feature_valid['convexity_valid'] = convexity is not None
            feature_valid['inertia_valid'] = inertia is not None
            feature_valid['num_segments_valid'] = num_segments is not None
            feature_valid['num_junctions_valid'] = True
            # Ensure all required features are present, set to None if missing
            for feat in required_features:
                if feat not in node_data:
                    node_data[feat] = None
            # Ensure all required valid flags are present, set to False if missing
            for flag in required_valid_flags:
                if flag not in feature_valid:
                    feature_valid[flag] = False
            node_data['feature_valid'] = feature_valid
            logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} features computed from LOGO vertices.")
        else:
            # Invalid polygon (bad geometry)
            node_data['geometry_valid'] = False
            # Set all required features to None
            for feat in required_features:
                node_data[feat] = None
            node_data['fallback_geometry'] = True
            # Set all required valid flags to False
            feature_valid = {k: False for k in required_valid_flags}
            node_data['feature_valid'] = feature_valid
            logging.warning(f"Invalid LOGO polygon for node {node_data.get('id', 'unknown')}, vertices: {vertices}")
    elif object_type == 'line' and vertices and len(vertices) == 2:
        # Proxy features for lines
        arr = np.array(vertices)
        midpoint = np.mean(arr, axis=0)
        node_data['centroid'] = midpoint.tolist()
        node_data['cx'] = midpoint[0]
        node_data['cy'] = midpoint[1]
        node_data['area'] = None
        node_data['perimeter'] = float(np.linalg.norm(arr[1] - arr[0]))
        node_data['orientation'] = float(np.degrees(np.arctan2(arr[1][1]-arr[0][1], arr[1][0]-arr[0][0])))
        node_data['aspect_ratio'] = None
        node_data['curvature'] = None
        node_data['skeleton_length'] = node_data['perimeter']
        node_data['symmetry_axis'] = None
        node_data['fallback_geometry'] = True
        node_data['bbox'] = [float(np.min(arr[:,0])), float(np.min(arr[:,1])), float(np.max(arr[:,0])), float(np.max(arr[:,1]))]
        # Set all required features to None if missing
        for feat in required_features:
            if feat not in node_data:
                node_data[feat] = None
        feature_valid = {
            'centroid_valid': True,
            'perimeter_valid': True,
            'orientation_valid': True,
            'skeleton_length_valid': True,
            'area_valid': False,
            'aspect_ratio_valid': False,
            'curvature_valid': False,
            'symmetry_axis_valid': False,
            'compactness_valid': False,
            'convexity_valid': False,
            'inertia_valid': False,
            'num_segments_valid': False,
            'num_junctions_valid': False
        }
        # Ensure all required valid flags are present, set to False if missing
        for flag in required_valid_flags:
            if flag not in feature_valid:
                feature_valid[flag] = False
        node_data['geometry_valid'] = True
        node_data['feature_valid'] = feature_valid
        logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} features computed for line.")
    else:
        # Degenerate or unsupported
        node_data['geometry_valid'] = False
        # Try to assign a fallback centroid if possible
        centroid = None
        if vertices and len(vertices) >= 2:
            try:
                arr = np.array(vertices)
                centroid = np.mean(arr, axis=0)
                node_data['centroid'] = centroid.tolist()
                node_data['cx'] = float(centroid[0])
                node_data['cy'] = float(centroid[1])
                logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} assigned fallback centroid {centroid.tolist()} for degenerate shape.")
            except Exception as e:
                node_data['centroid'] = None
                node_data['cx'] = None
                node_data['cy'] = None
                logging.warning(f"compute_physics_attributes: Fallback centroid computation failed for node {node_data.get('id', 'unknown')}: {e}")
        else:
            node_data['centroid'] = None
            node_data['cx'] = None
            node_data['cy'] = None
            logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} has no vertices for centroid assignment.")
        node_data['fallback_geometry'] = True
        # Set all required features to None
        for feat in required_features:
            if feat not in node_data:
                node_data[feat] = None
        # Set all required valid flags to False
        feature_valid = {k: False for k in required_valid_flags}
        # If we assigned a fallback centroid, mark centroid_valid True
        if centroid is not None:
            feature_valid['centroid_valid'] = True
        node_data['feature_valid'] = feature_valid
        logging.warning(f"Node {node_data.get('id', 'unknown')} missing or insufficient vertices for LOGO polygon or line.")
# --- Visualization Utility ---
def visualize_shape_and_centroid(vertices, centroid, save_path=None):
    import matplotlib.pyplot as plt
    arr = np.array(vertices)
    plt.figure(figsize=(5,5))
    if len(arr) > 0:
        plt.plot(arr[:,0], arr[:,1], 'o-', label='Vertices')
    if centroid is not None:
        plt.plot(centroid[0], centroid[1], 'rx', markersize=12, label='Centroid')
    plt.legend()
    plt.title('Shape and Centroid')
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

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
