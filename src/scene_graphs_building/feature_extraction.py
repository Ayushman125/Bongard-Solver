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
        # Automatically save visualization for this fallback case
        try:
            from src.scene_graphs_building.visualization import save_fallback_centroid_visualizations
            # Only save if a new sample was added
            if len(compute_physics_attributes.fallback_samples) > 0:
                save_fallback_centroid_visualizations(
                    [compute_physics_attributes.fallback_samples[-1]],
                    'visualizations/physics_fallbacks')
        except Exception as e:
            logging.warning(f"Could not save fallback centroid visualization: {e}")
        return
    # Always update centroid field to match computed cx/cy (even in fallback)
    node_data['centroid'] = [node_data.get('cx', 0.0), node_data.get('cy', 0.0)]
    # Automatically save visualization for this node (fallback case)
    try:
        from src.scene_graphs_building.visualization import save_fallback_centroid_visualizations
        save_fallback_centroid_visualizations(
            [{
                'id': node_data.get('id', 'unknown'),
                'vertices': node_data.get('vertices', []),
                'centroid': node_data['centroid']
            }],
            'visualizations/physics_fallbacks'
        )
    except Exception as e:
        logging.warning(f"Could not save centroid visualization: {e}")

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
    # --- Curvature Calculation ---
    curvature = 0.0
    if vertices and len(vertices) >= 3:
        # Estimate mean absolute curvature at vertices
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

    # --- Skeleton Length Calculation ---
    skeleton_length = 0.0
    if 'mask' in node_data:
        try:
            from skimage.morphology import skeletonize
            mask = node_data['mask']
            skel = skeletonize(mask > 0)
            # Count skeleton pixels and scale by pixel size
            skeleton_length = float(np.sum(skel))
        except Exception as e:
            logging.warning(f"Skeleton length computation failed: {e}")
    node_data['skeleton_length'] = skeleton_length

    # --- Symmetry Axis Calculation ---
    symmetry_axis = None
    try:
        # Use PCA major axis as symmetry axis estimate
        if vertices and len(vertices) >= 2:
            arr = np.array(vertices)
            cov_matrix = np.cov(arr.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            major_axis_idx = np.argmax(eigenvalues)
            principal_axis = eigenvectors[:, major_axis_idx]
            centroid = np.mean(arr, axis=0)
            # Represent axis as (point, direction)
            symmetry_axis = {
                'centroid': centroid.tolist(),
                'direction': principal_axis.tolist()
            }
    except Exception as e:
        logging.warning(f"Symmetry axis computation failed: {e}")
    node_data['symmetry_axis'] = symmetry_axis

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
    if not hasattr(compute_physics_attributes, 'fallback_count'):
        compute_physics_attributes.fallback_count = 0
        compute_physics_attributes.fallback_samples = []
    if not valid_poly:
        compute_physics_attributes.fallback_count += 1
        logging.warning(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} has invalid geometry and cannot be repaired. Fallback centroid will be used.")
        vertices = node_data.get('vertices')
        if vertices and len(vertices) >= 1:
            arr = np.array(vertices)
            mean_centroid = np.mean(arr, axis=0)
            node_data['cx'] = float(mean_centroid[0])
            node_data['cy'] = float(mean_centroid[1])
            node_data['area'] = 0.0
            node_data['perimeter'] = 0.0
            node_data['bbox'] = [float(np.min(arr[:,0])), float(np.min(arr[:,1])), float(np.max(arr[:,0])), float(np.max(arr[:,1]))]
            node_data['aspect_ratio'] = 1.0
            node_data['orientation'] = 0.0
            node_data['compactness'] = 0.0
            node_data['inertia'] = 0.0
            node_data['convexity'] = 0.0
            node_data['num_segments'] = 0
            node_data['num_junctions'] = 0
            logging.info(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} fallback centroid: {mean_centroid}")
            # Save sample for inspection (up to 10)
            if len(compute_physics_attributes.fallback_samples) < 10:
                compute_physics_attributes.fallback_samples.append({
                    'id': node_data.get('id', 'unknown'),
                    'vertices': vertices,
                    'centroid': mean_centroid.tolist()
                })
        return

    # Only run polygon feature computation if valid_poly is True
    if valid_poly:
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
            # MultiPolygon: aggregate features from all valid sub-polygons, log validity
            total_area = 0.0
            total_perimeter = 0.0
            total_inertia = 0.0
            total_convexity = 0.0
            total_compactness = 0.0
            total_num_junctions = 0
            weighted_cx = 0.0
            weighted_cy = 0.0
            valid_subpoly_count = 0
            for idx, subpoly in enumerate(poly.geoms):
                if not subpoly.is_valid or subpoly.area <= 0:
                    logging.warning(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} MultiPolygon subpoly {idx} invalid or zero area, skipping.")
                    continue
                valid_subpoly_count += 1
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
                # Area-weighted centroid
                weighted_cx += sub_features.get('cx', 0.0) * sub_area
                weighted_cy += sub_features.get('cy', 0.0) * sub_area
            if valid_subpoly_count == 0:
                node_data['cx'] = 0.0
                node_data['cy'] = 0.0
                logging.warning(f"compute_physics_attributes: Node {node_data.get('id', 'unknown')} MultiPolygon has no valid subpolygons, fallback centroid used.")
            else:
                node_data['cx'] = weighted_cx / total_area if total_area > 0 else 0.0
                node_data['cy'] = weighted_cy / total_area if total_area > 0 else 0.0
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
