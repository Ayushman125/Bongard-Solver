# --- Image Preprocessing for Robust Contour Extraction ---
def extract_clean_contours(mask, min_area=10, simplify_epsilon=2.0):
    """
    Given a binary mask (numpy array), extract clean, closed polygons (vertices) using:
    - Contour detection & hierarchy analysis
    - Hole-punching (internal contours)
    - Aggressive thinning & skeletonization
    - Morpholog                except Exception as e:
                    # Fallback to multipoint centroid calculation
                    vertices_array = np.array(vertices)l cleaning
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
    # Broaden geometry_valid: allow polygons, lines, arcs, points, and motifs as valid geometry
    if obj.get('object_type') in ('polygon', 'line', 'arc', 'point', 'motif'):
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

def _set_default_attributes(node_data, node_id):
    """Set default attributes for nodes with invalid or missing geometry."""
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
    
    # Set all features to None/default values
    for feat in required_features:
        if feat not in node_data:
            node_data[feat] = None
    
    # Set all validity flags to False
    feature_valid = {k: False for k in required_valid_flags}
    node_data['feature_valid'] = feature_valid
    node_data['geometry_valid'] = False
    node_data['fallback_geometry'] = True
    
    # Set default centroid to (0, 0) if not already set
    if node_data.get('centroid') is None:
        node_data['centroid'] = [0.0, 0.0]
        node_data['cx'] = 0.0
        node_data['cy'] = 0.0
        node_data['feature_valid']['centroid_valid'] = True
        logging.info(f"compute_physics_attributes: Node {node_id} assigned default centroid [0.0, 0.0] due to invalid geometry")

def compute_physics_attributes(node_data):
    """
    Computes robust physics attributes and asserts their domain validity.
    Now also computes perimeter and compactness.
    """
    # Initialize feature_valid dictionary early to avoid KeyError
    required_valid_flags = [
        'area_valid', 'aspect_ratio_valid', 'orientation_valid', 'curvature_valid',
        'perimeter_valid', 'centroid_valid', 'horizontal_asymmetry_valid', 'vertical_asymmetry_valid',
        'apex_x_position_valid', 'is_highly_curved_valid', 'compactness_valid', 'convexity_valid',
        'inertia_valid', 'num_segments_valid', 'num_junctions_valid'
    ]
    feature_valid = {k: False for k in required_valid_flags}
    node_data['feature_valid'] = feature_valid
    
    # --- Simplified, robust LOGO/NVLabs geometry handling ---
    vertices = node_data.get('vertices', [])
    object_type = node_data.get('object_type', None)
    is_closed = node_data.get('is_closed', False)
    action_program = node_data.get('action_program', [])
    # Enhanced node ID detection for better debugging
    node_id = None
    # Try multiple possible ID fields in order of preference
    for id_field in ['object_id', 'id', 'node_id', 'parent_shape_id']:
        if id_field in node_data and node_data[id_field] is not None:
            node_id = str(node_data[id_field])
            break
    
    # If still no ID found, try to construct one from available data
    if node_id is None or node_id == 'None':
        action_cmd = node_data.get('action_command', '')
        action_idx = node_data.get('action_index', '')
        if action_cmd and action_idx is not None:
            node_id = f"action_{action_idx}_{action_cmd.split('_')[0] if '_' in action_cmd else action_cmd}"
        else:
            node_id = 'unknown_node'
    
    # Log when we encounter problematic nodes for debugging
    if node_id == 'unknown_node' or 'unknown' in node_id:
        logging.warning(f"Node missing proper ID fields. Available keys: {list(node_data.keys())[:10]}...")
    
    # Enhanced vertex validation with fallback options
    if not vertices or len(vertices) == 0:
        # Try alternative vertex sources
        fallback_vertices = None
        for vertex_field in ['original_vertices', 'endpoints', 'centroid']:
            fallback_data = node_data.get(vertex_field, None)
            if fallback_data and len(fallback_data) >= 2:
                try:
                    # Handle different formats
                    if vertex_field == 'centroid' and len(fallback_data) == 2:
                        # Use centroid as single point
                        fallback_vertices = [fallback_data]
                        break
                    elif vertex_field == 'endpoints' and len(fallback_data) >= 4:
                        # Convert endpoints [x1,y1,x2,y2] to [[x1,y1],[x2,y2]]
                        fallback_vertices = [[fallback_data[0], fallback_data[1]], [fallback_data[2], fallback_data[3]]]
                        break
                    elif len(fallback_data) >= 2:
                        fallback_vertices = fallback_data
                        break
                except Exception:
                    continue
        
        if fallback_vertices:
            vertices = fallback_vertices
            logging.info(f"compute_physics_attributes: Node {node_id} using fallback vertices from alternative source")
        else:
            logging.warning(f"compute_physics_attributes: Node {node_id} has no vertices, skipping physics computation")
            _set_default_attributes(node_data, node_id)
            return node_data
    
    # Ensure vertices are valid numeric coordinates
    try:
        vertices_array = np.array(vertices, dtype=float)
        if vertices_array.size == 0 or vertices_array.ndim != 2 or vertices_array.shape[1] != 2:
            logging.warning(f"compute_physics_attributes: Node {node_id} has invalid vertex structure: {vertices}")
            _set_default_attributes(node_data, node_id)
            return node_data
    except (ValueError, TypeError) as e:
        logging.warning(f"compute_physics_attributes: Node {node_id} has non-numeric vertices: {e}")
        _set_default_attributes(node_data, node_id)
        return node_data
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
    
    # Validate geometry - allow polygons, lines, and motifs for proper geometric calculations
    # Polygons: require closed shape with 3+ vertices
    # Lines: require 2+ vertices for valid line segments
    # Motifs: require 2+ vertices for aggregate geometry
    geometry_valid = (
        (object_type == 'polygon' and is_closed and vertices and len(vertices) >= 3) or
        (object_type == 'line' and vertices and len(vertices) >= 2) or
        (object_type == 'motif' and vertices and len(vertices) >= 2)
    )
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
        if object_type == 'polygon' and is_closed and len(vertices) >= 3:
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
            else:
                # Invalid polygon - use fallback
                geometry_valid = False
        elif object_type == 'line' and len(vertices) >= 2:
            # Handle line objects with LineString
            from shapely.geometry import LineString
            line = LineString(vertices)
            if line.is_valid and line.length > 0:
                centroid = [float(line.centroid.x), float(line.centroid.y)]
                node_data['centroid'] = centroid
                node_data['cx'] = centroid[0]
                node_data['cy'] = centroid[1]
                node_data['fallback_geometry'] = False
                # Compute basic features for lines
                node_data['area'] = 0.0  # Lines have no area
                node_data['perimeter'] = line.length
                bounds = line.bounds
                node_data['bbox'] = [bounds[0], bounds[1], bounds[2], bounds[3]]
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                node_data['aspect_ratio'] = width / height if height > 0 else float('inf')
                # Line-specific calculations
                compactness = None  # Not applicable for lines
                convexity = None   # Not applicable for lines
                inertia = None     # Not applicable for lines
            else:
                # Invalid line - use fallback
                geometry_valid = False
        elif object_type == 'motif' and len(vertices) >= 2:
            # Handle motif objects - try polygon first, then multipoint/line
            from shapely.geometry import LineString, MultiPoint
            
            # Try to create a polygon from motif vertices if we have enough points
            if len(vertices) >= 3:
                try:
                    poly = Polygon(vertices)
                    if poly.is_valid and poly.area > 0:
                        # Valid polygon motif
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
                    else:
                        # Invalid polygon, try as multipoint
                        raise ValueError("Invalid polygon, trying multipoint")
                except Exception:
                    # Fallback to multipoint centroid calculation
                    vertices_array = np.array(vertices)
                    centroid = np.mean(vertices_array, axis=0)
                    node_data['centroid'] = centroid.tolist()
                    node_data['cx'] = float(centroid[0])
                    node_data['cy'] = float(centroid[1])
                    node_data['fallback_geometry'] = False
                    # Basic motif features - area from convex hull
                    try:
                        multipoint = MultiPoint(vertices)
                        convex_hull = multipoint.convex_hull
                        if hasattr(convex_hull, 'area') and convex_hull.area > 0:
                            node_data['area'] = float(convex_hull.area)
                            node_data['perimeter'] = float(convex_hull.length)
                        else:
                            node_data['area'] = 0.0
                            node_data['perimeter'] = 0.0
                    except Exception:
                        node_data['area'] = 0.0
                        node_data['perimeter'] = 0.0
                    
                    bounds = [float(np.min(vertices_array[:,0])), float(np.min(vertices_array[:,1])), 
                             float(np.max(vertices_array[:,0])), float(np.max(vertices_array[:,1]))]
                    node_data['bbox'] = bounds
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    node_data['aspect_ratio'] = width / height if height > 0 else 1.0
                    # Motif-specific defaults
                    compactness = None
                    convexity = None
                    inertia = None
            else:
                # Only 2 vertices - treat as line
                line = LineString(vertices)
                if line.is_valid and line.length > 0:
                    centroid = [float(line.centroid.x), float(line.centroid.y)]
                    node_data['centroid'] = centroid
                    node_data['cx'] = centroid[0]
                    node_data['cy'] = centroid[1]
                    node_data['fallback_geometry'] = False
                    node_data['area'] = 0.0
                    node_data['perimeter'] = line.length
                    bounds = line.bounds
                    node_data['bbox'] = [bounds[0], bounds[1], bounds[2], bounds[3]]
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    node_data['aspect_ratio'] = width / height if height > 0 else float('inf')
                    compactness = None
                    convexity = None
                    inertia = None
                else:
                    # Invalid motif - use fallback
                    geometry_valid = False
        else:
            # Unknown geometry type - use fallback
            geometry_valid = False
        
        # Continue with common processing for valid geometries
        if geometry_valid and not node_data.get('fallback_geometry', True):
            action_program = node_data.get('action_program', [])
            if (
                isinstance(action_program, list)
                and all(isinstance(cmd, dict) for cmd in action_program)
            ):
                num_segments = len(extract_line_segments(action_program))
            else:
                num_segments = None
            
            # Handle geometry-specific processing
            if object_type == 'polygon' and is_closed:
                coords_counter = Counter(tuple(c) for c in poly.exterior.coords)
                num_junctions = sum(1 for count in coords_counter.values() if count > 1)
                node_data.update({
                    'inertia': inertia,
                    'convexity': convexity,
                    'compactness': compactness,
                    'num_segments': int(num_segments) if num_segments is not None else None,
                    'num_junctions': int(num_junctions),
                })
            elif object_type == 'line':
                # For lines, set appropriate default values
                node_data.update({
                    'inertia': inertia,
                    'convexity': convexity,
                    'compactness': compactness,
                    'num_segments': int(num_segments) if num_segments is not None else None,
                    'num_junctions': 2,  # Lines have 2 endpoints
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
        if geometry_valid:
            # Only check poly validity for polygon objects - motifs and lines use different validation
            if object_type == 'polygon' and 'poly' in locals() and poly.is_valid and poly.area > 0:
                feature_valid['centroid_valid'] = True
            elif object_type in ['motif', 'line'] and node_data.get('centroid') is not None:
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
            # Populate symmetry_axis and predicate using parse_action_command from process_single_problem.py
            action_program = node_data.get('action_program', [])
            if isinstance(action_program, list) and all(isinstance(cmd, str) for cmd in action_program):
                from src.scene_graphs_building.process_single_problem import parse_action_command
                # Example: extract symmetry_axis and predicate from parsed commands
                predicates = []
                symmetry_axes = []
                for cmd in action_program:
                    parsed = parse_action_command(cmd)
                    if parsed:
                        if parsed.get('type') == 'line':
                            predicates.append('line')
                            # For lines, symmetry axis is the direction
                            symmetry_axes.append(parsed.get('mode'))
                        elif parsed.get('type') == 'arc':
                            predicates.append('arc')
                            symmetry_axes.append(parsed.get('mode'))
                node_data['predicate'] = predicates[0] if predicates else None
                node_data['symmetry_axis'] = symmetry_axes[0] if symmetry_axes else None
            node_data['feature_valid'] = feature_valid
            logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} features computed from LOGO vertices.")
        else:
            # Invalid polygon (bad geometry)
            node_data['geometry_valid'] = False
            for feat in required_features:
                node_data[feat] = None
            node_data['fallback_geometry'] = True
            feature_valid = {k: False for k in required_valid_flags}
            node_data['feature_valid'] = feature_valid
            node_data['symmetry_axis'] = None
            node_data['predicate'] = None
            logging.warning(f"Invalid LOGO polygon for node {node_data.get('id', 'unknown')}, vertices: {vertices}")
            for feat in required_features:
                if feat not in node_data:
                    node_data[feat] = None
            # Ensure all required valid flags are present, set to False if missing
            for flag in required_valid_flags:
                if flag not in feature_valid:
                    feature_valid[flag] = False
            node_data['feature_valid'] = feature_valid
            logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} features computed from LOGO vertices.")
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
        # Degenerate or unsupported geometry - use more robust fallback logic
        node_data['geometry_valid'] = False
        
        # Try to assign a fallback centroid if possible
        centroid = None
        if vertices and len(vertices) >= 1:
            try:
                vertices_array = np.array(vertices, dtype=float)
                if vertices_array.size >= 2 and vertices_array.ndim >= 1:
                    # Handle both [[x,y], [x,y]] and [x,y,x,y] formats
                    if vertices_array.ndim == 1 and len(vertices_array) >= 2:
                        # Reshape flat array to coordinate pairs
                        if len(vertices_array) % 2 == 0:
                            vertices_array = vertices_array.reshape(-1, 2)
                        else:
                            vertices_array = vertices_array[:-1].reshape(-1, 2)  # Drop last odd element
                    
                    if vertices_array.ndim == 2 and vertices_array.shape[1] == 2:
                        centroid = np.mean(vertices_array, axis=0)
                        node_data['centroid'] = centroid.tolist()
                        node_data['cx'] = float(centroid[0])
                        node_data['cy'] = float(centroid[1])
                        node_data['feature_valid']['centroid_valid'] = True
                        logging.info(f"compute_physics_attributes: Node {node_id} assigned fallback centroid {centroid.tolist()} from vertices")
                    else:
                        logging.warning(f"compute_physics_attributes: Node {node_id} has invalid vertex array shape: {vertices_array.shape}")
                        _set_default_attributes(node_data, node_id)
                else:
                    logging.warning(f"compute_physics_attributes: Node {node_id} has insufficient vertex data: {vertices_array}")
                    _set_default_attributes(node_data, node_id)
            except Exception as e:
                logging.warning(f"compute_physics_attributes: Failed to compute fallback centroid for node {node_id}: {e}")
                _set_default_attributes(node_data, node_id)
        else:
            logging.warning(f"compute_physics_attributes: Node {node_id} has no valid vertices for centroid computation")
            _set_default_attributes(node_data, node_id)
        
        node_data['fallback_geometry'] = True
        
        # Set all other required features to None if not set by fallback logic
        for feat in required_features:
            if feat not in node_data:
                node_data[feat] = None
        
        # Set all required valid flags to False except those set by fallback logic
        for flag in required_valid_flags:
            if flag not in node_data.get('feature_valid', {}):
                if 'feature_valid' not in node_data:
                    node_data['feature_valid'] = {}
                node_data['feature_valid'][flag] = False
    
    # Ensure the function always returns the node_data
    return node_data

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
