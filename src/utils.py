# Folder: bongard_solver/src/
# File: utils.py

import torch
import numpy as np
import random
import os
import logging
import sys
import json
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from numba import njit
from functools import lru_cache
import torch.nn as nn

# Conditional import for torch_geometric.nn.global_mean_pool
HAS_PYG_POOL = False
try:
    from torch_geometric.nn import global_mean_pool
    HAS_PYG_POOL = True
    logger = logging.getLogger(__name__)
    logger.info("torch_geometric.nn.global_mean_pool found and enabled.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch Geometric not found. global_mean_pool will be a dummy function.")
    # Dummy global_mean_pool if PyG is not available
    def global_mean_pool(node_embeds: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Dummy global mean pooling function when PyTorch Geometric is not available.
        Performs mean pooling per graph based on the batch index.
        """
        if node_embeds.numel() == 0:
            # Return an empty tensor with the correct feature dimension if no nodes
            return torch.empty(0, node_embeds.shape[-1], device=node_embeds.device)
        unique_batches = torch.unique(batch)
        pooled_embeddings = []
        for b_id in unique_batches:
            mask = (batch == b_id)
            if node_embeds[mask].numel() > 0:
                pooled_embeddings.append(torch.mean(node_embeds[mask], dim=0, keepdim=True))
            else:
                # Handle empty graphs if necessary, e.g., return zeros
                pooled_embeddings.append(torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device))
        return torch.cat(pooled_embeddings, dim=0) if pooled_embeddings else torch.empty(0, node_embeds.shape[-1], device=node_embeds.device)

logger = logging.getLogger(__name__)

# Forward declaration for type hinting if Workspace is in another file
if TYPE_CHECKING:
    from src.emergent.workspace_ext import Workspace  # Corrected import path

# Import augment_image from src/utils/augment.py
try:
    from src.utils.augment import augment_image
    HAS_AUGMENT_IMAGE = True
except ImportError:
    logger.warning("Could not import augment_image from src/utils/augment.py. Image augmentations will not be applied via this utility.")
    HAS_AUGMENT_IMAGE = False
    # Dummy augment_image function
    def augment_image(img_np: np.ndarray) -> np.ndarray:
        logger.warning("Using dummy augment_image: returns original image.")
        return img_np


# --- Logging Setup ---
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Sets up the logging configuration for the project.
    Args:
        log_level (str): The minimum level of messages to log (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (Optional[str]): Path to a file where logs should also be written.
                                   If None, logs only to console.
    """
    # Remove all existing handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
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
    import yaml  # Import yaml here to avoid circular dependency if config.py imports utils
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}.")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

# 11.1 Config-Sanity Checker
def check_unused_flags(cfg: Dict[str, Any], used_keys: List[str]):
    """
    Checks for unused flags in the configuration dictionary.
    Logs a warning if any flags are found that are not in the `used_keys` list.
    This function performs a shallow check on the top-level keys.
    For nested dictionaries, you would need to implement a recursive check
    or pass nested keys explicitly.
    Args:
        cfg (Dict[str, Any]): The configuration dictionary.
        used_keys (List[str]): A list of top-level keys that are expected to be used.
    """
    unused = set(cfg.keys()) - set(used_keys)
    if unused:
        logging.warning(f"Unused config flags detected: {unused}. Consider reviewing your config or `used_keys` list.")
    else:
        logging.info("All top-level config flags appear to be used.")

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
    # Assuming these are defined in a config module that is accessible
    try:
        # Relative import from the project root's config.py
        from ..config import ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP, \
                             ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, \
                             RELATION_MAP
    except ImportError:
        logger.warning("Could not import attribute/relation maps from config. Using dummy values.")
        ATTRIBUTE_SHAPE_MAP = {'circle':0, 'square':1}
        ATTRIBUTE_COLOR_MAP = {'red':0, 'blue':1}
        ATTRIBUTE_FILL_MAP = {'filled':0, 'outlined':1}
        ATTRIBUTE_SIZE_MAP = {'small':0, 'medium':1}
        ATTRIBUTE_ORIENTATION_MAP = {'horizontal':0}
        ATTRIBUTE_TEXTURE_MAP = {'smooth':0}
        RELATION_MAP = {'left_of':0, 'above':1}

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

@lru_cache(maxsize=128)
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
            if i == j: continue     # Skip self-loops
            edge_index_map[(i, j)] = idx
            idx += 1
    return edge_index_map

def get_predicted_relation(
    objects: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    bbox_iou_threshold: float = 0.1,
    spatial_tolerance_ratio: float = 0.1
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
            bbox1 = obj1['bbox_xyxy'] # Ensure bbox is xyxy
            bbox2 = obj2['bbox_xyxy']
            
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
            elif iou > 0:    # Small overlap, but not significant enough for 'overlapping'
                inferred_relations.append({'type': 'touching', 'subject_id': obj1['id'], 'object_id': obj2['id']})
                inferred_relations.append({'type': 'touching', 'subject_id': obj2['id'], 'object_id': obj1['id']})
            elif iou == 0 and (
                abs(bbox1[2] - bbox2[0]) < spatial_tolerance_ratio * image_width or     # right edge of 1 near left edge of 2
                abs(bbox2[2] - bbox1[0]) < spatial_tolerance_ratio * image_width or     # right edge of 2 near left edge of 1
                abs(bbox1[3] - bbox2[1]) < spatial_tolerance_ratio * image_height or     # bottom edge of 1 near top edge of 2
                abs(bbox2[3] - bbox1[1]) < spatial_tolerance_ratio * image_height       # bottom edge of 2 near top edge of 1
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

def cross_attend(query: torch.Tensor, context: torch.Tensor, embed_dim: int, num_heads: int = 1) -> torch.Tensor:
    """
    Performs cross-attention between a query and a context.
    Args:
        query (torch.Tensor): The query tensor (Batch_size, Query_dim).
                              This will be unsqueezed to (Batch_size, 1, Query_dim) for attention.
        context (torch.Tensor): The context tensor (Batch_size, Context_dim).
                                This will be unsqueezed to (Batch_size, 1, Context_dim) for attention.
        embed_dim (int): The embedding dimension for the attention mechanism.
                         This should typically match the `Query_dim` of the query.
        num_heads (int): Number of attention heads.
    Returns:
        torch.Tensor: The output of the cross-attention (Batch_size, Query_dim).
    """
    if query.dim() == 1:     # Handle single query
        query = query.unsqueeze(0)
    if context.dim() == 1:     # Handle single context
        context = context.unsqueeze(0)
    
    query_seq = query.unsqueeze(1)       # (B, 1, Query_dim)
    context_seq = context.unsqueeze(1)   # (B, 1, Context_dim)
    
    attn = nn.MultiheadAttention(
        embed_dim=query.shape[-1],       # Query_dim
        num_heads=num_heads,
        kdim=context.shape[-1],          # Context_dim
        vdim=context.shape[-1],          # Context_dim
        batch_first=True                 # Input/output tensors are (batch, seq_len, feature)
    ).to(query.device)
    
    attn_output, _ = attn(
        query=query_seq,
        key=context_seq,
        value=context_seq
    )
    
    return attn_output.squeeze(1)

# --- Dummy Forward Feature-Dim Inference ---
def infer_feature_dim(model: nn.Module, img_size: int, device: torch.device) -> int:
    """
    Infers the output feature dimension of a model by performing a dummy forward pass.
    This is useful for dynamically setting input dimensions for subsequent layers.
    Args:
        model (nn.Module): The model whose feature dimension needs to be inferred.
        img_size (int): The expected input image size (height and width).
        device (torch.device): The device to perform the dummy forward pass on.
    Returns:
        int: The flattened output feature dimension of the model.
    """
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    
    with torch.no_grad():
        fmap_or_list = model(x)
        
        if isinstance(fmap_or_list, list):
            fmap = fmap_or_list[-1]
        else:
            fmap = fmap_or_list
        
        if fmap.ndim == 4:
            return fmap.view(fmap.size(0), -1).size(1)
        elif fmap.ndim == 3:
            return fmap.view(fmap.size(0), -1).size(1)
        elif fmap.ndim == 2:
            return fmap.size(1)
        else:
            raise ValueError(f"Unsupported feature map dimension for inference: {fmap.ndim}")

# --- Graph Pooling Helper ---
def graph_pool(node_embeds: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    Wraps torch_geometric.nn.global_mean_pool for consistent graph pooling.
    Args:
        node_embeds (torch.Tensor): Node embeddings (N_nodes, D_features).
        batch_idx (torch.Tensor): Batch assignment for each node (N_nodes,).
                                   Indicates which graph each node belongs to.
    Returns:
        torch.Tensor: Global graph embeddings (N_graphs, D_features).
    """
    if not HAS_PYG_POOL:
        logger.warning("PyTorch Geometric global_mean_pool not available. Using dummy mean pooling.")
        unique_batches = torch.unique(batch_idx)
        pooled_embeddings = []
        for b_id in unique_batches:
            mask = (batch_idx == b_id)
            if node_embeds[mask].numel() > 0:
                pooled_embeddings.append(torch.mean(node_embeds[mask], dim=0, keepdim=True))
            else:
                pooled_embeddings.append(torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device))
        return torch.cat(pooled_embeddings, dim=0) if pooled_embeddings else torch.empty(0, node_embeds.shape[-1], device=node_embeds.device)
    
    return global_mean_pool(node_embeds, batch_idx)

# --- New: Temperature Utility ---
def compute_temperature(ws: 'Workspace', alpha: float = 0.7, beta: float = 0.3) -> float:
    """
    Computes the system temperature based on concept network activation and
    workspace structure coherence.
    Args:
        ws (Workspace): The current Workspace instance.
        alpha (float): Weight for the average concept activation component.
        beta (float): Weight for the structure coherence component.
    Returns:
        float: The computed temperature (between 0.0 and 1.0).
    """
    if not ws.concept_net.nodes:
        logger.warning("Concept network has no nodes. Returning default temperature 1.0.")
        return 1.0
    
    total_activation = sum(n.activation for n in ws.concept_net.nodes.values())
    avg_act = total_activation / len(ws.concept_net.nodes)
    
    coh = ws.structure_coherence()
    
    temp = alpha * (1 - avg_act) + beta * (1 - coh)
    
    temp = max(0.0, min(1.0, temp))
    logger.debug(f"Computed temperature: {temp:.4f} (avg_act: {avg_act:.4f}, coherence: {coh:.4f})")
    return temp

