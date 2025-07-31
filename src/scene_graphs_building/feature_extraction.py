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

    vertices = node_data.get('vertices')
    if not vertices or len(vertices) < 3:
        logging.warning(f"compute_physics_attributes: Invalid or missing vertices for node {node_data.get('id', 'unknown')}. Initializing with default physics attributes.")
        node_data.update({'area': 0.0, 'inertia': 0.0, 'convexity': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'perimeter': 0.0, 'orientation': 0.0, 'compactness': 0.0, 'num_segments': 0, 'num_junctions': 0})
        return

    poly = Polygon(vertices)
    
    # Compute basic features first
    basic_features = compute_basic_features(poly)
    node_data.update(basic_features)

    area = node_data['area'] # Use the computed area

    assert area >= 0.0, f"Negative area {area} for node {node_data.get('id')}"

    try:
        # Shapely's moment_of_inertia is about the centroid
        inertia = poly.moment_of_inertia
    except Exception:
        inertia = 0.0 # Default to 0 on error

    convexity = poly.convex_hull.area / area if area > 0 else 0.0

    # New attributes: compactness, num_segments, num_junctions
    perimeter = node_data['perimeter'] # Use computed perimeter
    compactness = perimeter**2 / (4 * math.pi * area) if area > 0 else 0.0

    action_program = node_data.get('action_program', [])
    num_segments = len(extract_line_segments(action_program))
    
    # Simple way to count junctions (points that appear more than once in exterior coords)
    num_junctions = 0
    if poly.exterior:
        coords_counter = Counter(tuple(c) for c in poly.exterior.coords)
        num_junctions = sum(1 for count in coords_counter.values() if count > 1)

    node_data.update({
        'inertia': float(inertia),
        'convexity': float(convexity),
        'compactness': float(compactness),
        'num_segments': int(num_segments),
        'num_junctions': int(num_junctions),
    })

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
