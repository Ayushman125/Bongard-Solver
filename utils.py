# Folder: bongard_solver/
# File: utils.py

import torch
import numpy as np
import random
import os
import logging
import sys # Added for setup_logging
import json
from typing import List, Dict, Any, Tuple, Optional
from numba import njit # For JIT compilation of Python functions
from functools import lru_cache # For caching

logger = logging.getLogger(__name__)

# --- Logging Setup ---
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Sets up the logging configuration for the project.
    Args:
        log_level (str): The minimum level of messages to log (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (Optional[str]): Path to a file where logs should also be written.
                                 If None, logs only to console.
    """
    # Ensure the root logger is configured only once
    if not logging.root.handlers:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        logging.root.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logging.root.addHandler(file_handler)
        
        logging.root.setLevel(numeric_level)
        logger.info(f"Logging configured with level: {log_level} and file: {log_file}")
    else:
        logger.debug("Logging already configured. Skipping setup.")

# --- Random Seed Setting ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}.")

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    import yaml # Import yaml here to avoid circular dependency if config.py imports utils
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

# --- IoU Calculation (JIT compiled) ---
@njit
def _calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) of two bounding boxes.
    Optimized with Numba for faster execution.
    Args:
        box1 (List[float]): Bounding box [x1, y1, x2, y2].
        box2 (List[float]): Bounding box [x1, y1, x2, y2].
    Returns:
        float: IoU value.
    """
    # Convert lists to NumPy arrays for Numba compatibility
    # Numba can directly work with Python lists of numbers, but explicit conversion
    # might sometimes help clarity or type inference for complex cases.
    # For simple lists of floats, it's often not strictly necessary.
    # box1_np = np.array(box1)
    # box2_np = np.array(box2)

    x1_inter = np.maximum(box1[0], box2[0])
    y1_inter = np.maximum(box1[1], box2[1])
    x2_inter = np.minimum(box1[2], box2[2])
    y2_inter = np.minimum(box1[3], box2[3])
    inter_area = np.maximum(0.0, x2_inter - x1_inter) * np.maximum(0.0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = float(box1_area + box2_area - inter_area)
    
    return inter_area / union_area if union_area > 0 else 0.0


# --- Scene Graph Utilities ---

def get_symbolic_embedding_dims(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Returns the expected embedding dimensions for each symbolic attribute.
    This function now takes the full config dict.
    """
    # Import config attributes locally to avoid circular imports if config.py imports utils
    from config import ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP, \
                       ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, \
                       RELATION_MAP
    
    dims = {
        'shape_dim': len(ATTRIBUTE_SHAPE_MAP),
        'color_dim': len(ATTRIBUTE_COLOR_MAP),
        'fill_dim': len(ATTRIBUTE_FILL_MAP),
        'size_dim': len(ATTRIBUTE_SIZE_MAP),
        'orientation_dim': len(ATTRIBUTE_ORIENTATION_MAP),
        'texture_dim': len(ATTRIBUTE_TEXTURE_MAP),
        'relation_dim': len(RELATION_MAP)
    }
    logger.debug(f"Symbolic embedding dimensions: {dims}")
    return dims

@lru_cache(maxsize=128) # Cache the mapping for common max_objects values
def make_edge_index_map(num_objects: int) -> Dict[Tuple[int, int], int]:
    """
    Creates a consistent mapping from (subject_id, object_id) pairs to a linear edge index.
    This is useful for converting graph relations to a flattened tensor for GNN output.
    Assumes a fully connected graph without self-loops.
    Args:
        num_objects (int): The number of objects in a scene graph.
    Returns:
        Dict[Tuple[int, int], int]: A dictionary mapping (subject_id, object_id) tuples to their
                                    corresponding linear index.
    """
    edge_index_map = {}
    idx = 0
    for i in range(num_objects):
        for j in range(num_objects):
            if i == j: continue   # Skip self-loops
            edge_index_map[(i, j)] = idx
            idx += 1
    return edge_index_map

def get_predicted_relation(
    objects: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    bbox_iou_threshold: float = 0.1,  # For 'overlapping' and 'touching'
    spatial_tolerance_ratio: float = 0.1  # For 'above', 'below', 'left_of', 'right_of'
) -> List[Dict[str, Any]]:
    """
    Infers relations between objects based on their attributes and spatial positions.
    This is a conceptual function and needs robust implementation based on your
    definition of relations and how they are derived from object properties.
    Args:
        objects (List[Dict[str, Any]]): A list of detected/ground truth objects,
                                        each with 'id', 'shape', 'color', 'bbox', etc.
        relations (List[Dict[str, Any]]): Existing relations (e.g., from ground truth).
                                          New relations might be added or existing
                                          ones validated.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        bbox_iou_threshold (float): IoU threshold to consider objects overlapping/touching.
        spatial_tolerance_ratio (float): Tolerance for spatial relations (e.g., how close
                                        Y-coords must be for 'above' to be true).
    Returns:
        List[Dict[str, Any]]: A list of inferred or validated relations.
    """
    inferred_relations = []
    object_map = {obj['id']: obj for obj in objects}
    
    # Iterate through all unique pairs of objects
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj1 = objects[i]
            obj2 = objects[j]
            
            # --- Attribute-based relations ---
            if obj1.get('attributes', {}).get('shape') == obj2.get('attributes', {}).get('shape'):
                inferred_relations.append({'type': 'same_shape', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'same_shape', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            else:
                inferred_relations.append({'type': 'different_shape', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'different_shape', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            
            if obj1.get('attributes', {}).get('color') == obj2.get('attributes', {}).get('color'):
                inferred_relations.append({'type': 'same_color', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'same_color', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            else:
                inferred_relations.append({'type': 'different_color', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'different_color', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            
            # For size, assuming 'small', 'medium', 'large' can be ordered
            size_order = {'small': 0, 'medium': 1, 'large': 2}
            if size_order.get(obj1.get('attributes', {}).get('size')) == size_order.get(obj2.get('attributes', {}).get('size')):
                inferred_relations.append({'type': 'same_size', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'same_size', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            else:
                inferred_relations.append({'type': 'different_size', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'different_size', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            
            # --- Spatial relations ---
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            
            # Calculate centers
            center1_x = bbox1[0] + (bbox1[2] - bbox1[0]) / 2
            center1_y = bbox1[1] + (bbox1[3] - bbox1[1]) / 2
            center2_x = bbox2[0] + (bbox2[2] - bbox2[0]) / 2
            center2_y = bbox2[1] + (bbox2[3] - bbox2[1]) / 2
            
            # Overlapping / Touching
            iou = _calculate_iou(bbox1, bbox2)
            if iou > bbox_iou_threshold:
                inferred_relations.append({'type': 'overlapping', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'overlapping', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif iou > 0:  # Small overlap, but not significant enough for 'overlapping'
                inferred_relations.append({'type': 'touching', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'touching', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif iou == 0 and (
                abs(bbox1[2] - bbox2[0]) < spatial_tolerance_ratio * image_width or  # right edge of 1 near left edge of 2
                abs(bbox2[2] - bbox1[0]) < spatial_tolerance_ratio * image_width or  # right edge of 2 near left edge of 1
                abs(bbox1[3] - bbox2[1]) < spatial_tolerance_ratio * image_height or  # bottom edge of 1 near top edge of 2
                abs(bbox2[3] - bbox1[1]) < spatial_tolerance_ratio * image_height     # bottom edge of 2 near top edge of 1
            ):
                # Check if bounding boxes are very close but not overlapping
                inferred_relations.append({'type': 'touching', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'touching', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            
            # Above/Below
            if center1_y < center2_y - spatial_tolerance_ratio * image_height:
                inferred_relations.append({'type': 'above', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'below', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif center2_y < center1_y - spatial_tolerance_ratio * image_height:
                inferred_relations.append({'type': 'above', 'subject_id': obj2['id'], 'object_id': obj1['id']})
                inferred_relations.append({'type': 'below', 'subject_id': obj1['id'], 'object_id': obj2['id']})
            
            # Left/Right
            if center1_x < center2_x - spatial_tolerance_ratio * image_width:
                inferred_relations.append({'type': 'left_of', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'right_of', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif center2_x < center1_x - spatial_tolerance_ratio * image_width:
                inferred_relations.append({'type': 'left_of', 'subject_id': obj2['id'], 'object_id': obj1['id']})
                inferred_relations.append({'type': 'right_of', 'subject_id': obj1['id'], 'object_id': obj2['id']})
            
            # Inside/Outside (more complex, requires checking if one bbox is fully contained in another)
            def is_bbox_inside(bbox_inner, bbox_outer):
                return bbox_inner[0] >= bbox_outer[0] and \
                       bbox_inner[1] >= bbox_outer[1] and \
                       bbox_inner[2] <= bbox_outer[2] and \
                       bbox_inner[3] <= bbox_outer[3]

            if is_bbox_inside(bbox1, bbox2):
                inferred_relations.append({'type': 'inside', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'outside', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif is_bbox_inside(bbox2, bbox1):
                inferred_relations.append({'type': 'inside', 'subject_id': obj2['id'], 'object_id': obj1['id']})
                inferred_relations.append({'type': 'outside', 'subject_id': obj1['id'], 'object_id': obj2['id']})
    
    logger.debug(f"Inferred {len(inferred_relations)} relations.")
    return inferred_relations

# --- Caching Notes ---
# Memory-Mapped Index Cache:
# For very large datasets where image paths or other metadata are stored in files,
# using numpy.load(filename, mmap_mode='r') can create a memory-mapped array.
# This allows accessing data from disk as if it were in memory, reducing RAM usage
# and improving load times for repeated access to large index files.
# This would typically be implemented in data loading functions (e.g., in data.py)
# when loading manifest files.
# Example:
# image_paths = np.load('image_manifest.npy', mmap_mode='r') # in data.py

# Fractal Cache with LRU:
# The `@lru_cache` decorator (from functools) is already used above for `make_edge_index_map`.
# If there were functions generating fractals or other complex procedural content
# that are deterministic for given inputs, `@lru_cache` would be applied to them
# to store results in memory and avoid recomputation for repeated inputs.
# Ensure `maxsize` is tuned to balance memory usage and caching effectiveness.

