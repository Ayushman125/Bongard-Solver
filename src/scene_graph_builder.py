# Folder: bongard_solver/src/scene_graph_builder.py

import numpy as np
import networkx as nx
import logging
import cv2 # For mask processing and contour detection
from PIL import Image # For opening images
import math # For geometric calculations
from collections import defaultdict
import json # For saving/loading symbolic annotations
from typing import List, Dict, Any, Tuple, Optional

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Helper Functions for Mask Processing (from symbolic_fusion.py) ---

def _mask_to_bbox(mask: np.ndarray) -> list:
    """
    Converts a binary mask to an axis-aligned bounding box [xmin, ymin, xmax, ymax].
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        list: Bounding box coordinates [xmin, ymin, xmax, ymax], or [0,0,0,0] if mask is empty.
    """
    if mask.ndim != 2:
        logger.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return [0, 0, 0, 0]

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0] # Return empty box if no foreground pixels

    xmin, ymin = int(xs.min()), int(ys.min())
    xmax, ymax = int(xs.max()), int(ys.max())
    return [xmin, ymin, xmax, ymax]

def _mask_centroid(mask: np.ndarray) -> list:
    """
    Calculates the centroid (mean x, mean y) of foreground pixels in a binary mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        list: Centroid coordinates [cx, cy], or [0,0] if mask is empty.
    """
    if mask.ndim != 2:
        logger.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return [0, 0]

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0] # Return [0,0] if no foreground pixels

    return [float(xs.mean()), float(ys.mean())]

def _mask_area(mask: np.ndarray) -> int:
    """
    Calculates the area (number of foreground pixels) of a binary mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        int: The area of the mask.
    """
    if mask.ndim != 2:
        logger.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return 0
    return int(np.sum(mask > 0))

def _mask_aspect_ratio(mask: np.ndarray) -> float:
    """
    Calculates the aspect ratio (width / height) of the bounding box of a mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        float: Aspect ratio, or 0.0 if mask is empty or height is zero.
    """
    bbox = _mask_to_bbox(mask)
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if height == 0:
        return 0.0 # Avoid division by zero
    return float(width / height)

def _mask_solidity(mask: np.ndarray) -> float:
    """
    Calculates the solidity of a mask (area / bounding box area).
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        float: Solidity, or 0.0 if mask or bbox is empty.
    """
    area = _mask_area(mask)
    bbox = _mask_to_bbox(mask)
    xmin, ymin, xmax, ymax = bbox
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area == 0:
        return 0.0
    return float(area / bbox_area)

# --- Geometric Relation Helpers (using xyxy bbox format) (from symbolic_fusion.py) ---
def iou_xyxy(bbox1: list, bbox2: list) -> float:
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes.
    Boxes are expected in [x0, y0, x1, y1] format (top-left, bottom-right corners).
    Args:
        bbox1 (list): Bounding box 1.
        bbox2 (list): Bounding box 2.
    Returns:
        float: IoU value.
    """
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def _get_spatial_relations(bbox1: list, bbox2: list, threshold_iou: float = 0.01,
                           center_dist_thresh_px: float = 10.0) -> list:
    """
    Determines common spatial relations between two bounding boxes.
    Args:
        bbox1 (list): [xmin, ymin, xmax, ymax] for object 1.
        bbox2 (list): [xmin, ymin, xmax, ymax] for object 2.
        threshold_iou (float): IoU threshold to consider boxes as "overlapping".
        center_dist_thresh_px (float): Pixel threshold for "aligned" or "near" relations.
    Returns:
        list: A list of strings representing detected relations.
    """
    relations = []
    
    # Calculate centroids
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2

    # Horizontal relations
    if cx1 < cx2 - center_dist_thresh_px:
        relations.append('left_of') # Changed to underscore for consistency with DSL
    elif cx1 > cx2 + center_dist_thresh_px:
        relations.append('right_of') # Changed to underscore for consistency with DSL
    
    # Vertical relations (smaller y is higher)
    if cy1 < cy2 - center_dist_thresh_px:
        relations.append('above') # Changed to underscore for consistency with DSL
    elif cy1 > cy2 + center_dist_thresh_px:
        relations.append('below') # Changed to underscore for consistency with DSL

    # Overlap
    current_iou = iou_xyxy(bbox1, bbox2)
    if current_iou > threshold_iou:
        relations.append('intersects') # Changed to underscore for consistency with DSL
        # Removed 'highly-overlaps' for simplicity and consistency with DSL
    
    # Contains (A contains B)
    # This is a strict bounding box containment. For mask-based containment,
    # you'd need to check if mask B is entirely within mask A.
    if (bbox1[0] <= bbox2[0] and bbox1[1] <= bbox2[1] and
        bbox1[2] >= bbox2[2] and bbox1[3] >= bbox2[3]):
        relations.append('contains') # Changed to underscore for consistency with DSL
    
    # Is contained by (A is contained by B)
    if (bbox2[0] <= bbox1[0] and bbox2[1] <= bbox1[1] and
        bbox2[2] >= bbox1[2] and bbox2[3] >= bbox1[3]):
        relations.append('is_contained_by') # New relation for clarity, or can be inferred from 'contains'

    # Alignment (horizontal or vertical)
    if abs(cx1 - cx2) < center_dist_thresh_px:
        relations.append('aligned_vertically') # Changed to underscore for consistency with DSL
    if abs(cy1 - cy2) < center_dist_thresh_px:
        relations.append('aligned_horizontally') # Changed to underscore for consistency with DSL

    return relations

# --- Clustering/Grouping Helpers (from symbolic_fusion.py) ---
def _cluster_by_proximity(G: nx.Graph, threshold: float = 50.0) -> dict:
    """
    Clusters nodes in a graph based on edge 'distance' attribute using connected components.
    Args:
        G (nx.Graph): Graph where nodes have 'centroid' properties and edges can have 'distance'.
        threshold (float): Maximum distance for two nodes to be considered in the same cluster.
    Returns:
        dict: A dictionary mapping node index to cluster ID.
    """
    clusters = {}
    cluster_id = 0
    visited = set()

    for node in G.nodes:
        if node in visited:
            continue

        # Start a BFS from the current unvisited node
        queue = [node]
        current_cluster_nodes = []

        while queue:
            n = queue.pop(0) # Use pop(0) for BFS (queue behavior)
            if n in visited:
                continue
            
            visited.add(n)
            current_cluster_nodes.append(n)
            clusters[n] = cluster_id

            for nbr in G.neighbors(n):
                # Ensure 'distance' attribute exists and is below threshold
                edge_data = G[n][nbr]
                if 'distance' in edge_data and edge_data['distance'] < threshold and nbr not in visited:
                    queue.append(nbr)
        
        if current_cluster_nodes: # Increment cluster_id only if a new cluster was found
            cluster_id += 1
    
    return clusters

# --- Object Detection/Segmentation (Placeholder for external models) ---
class ClassicalCVCropper:
    """
    A classical computer vision approach to object detection and mask extraction
    using contour detection. Replaces Mask R-CNN dependency.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contour_conf_thresh = config.get('cv_contour_conf_thresh', 0.1) # Min area ratio for a contour to be considered an object
        logger.info(f"ClassicalCVCropper initialized with contour_conf_thresh: {self.contour_conf_thresh}")

    def detect_objects(self, image_np: np.ndarray) -> List[np.ndarray]:
        """
        Detects objects in an image using contour detection and returns their masks.
        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, 3).
        Returns:
            List[np.ndarray]: A list of binary masks, one for each detected object.
        """
        if image_np is None or image_np.size == 0:
            logger.warning("Input image is empty or None for object detection.")
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise and help contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection or adaptive thresholding
        # For simplicity, let's use a fixed threshold or Otsu's method
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []
        image_area = image_np.shape[0] * image_np.shape[1]
        
        for contour in contours:
            # Filter small contours (noise)
            area = cv2.contourArea(contour)
            if area > image_area * self.contour_conf_thresh: # Only consider contours above a certain relative area
                mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                masks.append((mask > 0).astype(np.uint8)) # Convert to binary 0/1 mask
        
        logger.info(f"Detected {len(masks)} objects using ClassicalCVCropper.")
        return masks

# --- Main Scene Graph Builder Class ---

class SceneGraphBuilder:
    """
    Constructs a symbolic scene graph from an image.
    This involves object detection, attribute extraction, and relation inference.
    """
    def __init__(self, images: List[Any], config: Dict[str, Any]):
        """
        Args:
            images (List[Any]): List of image data (e.g., file paths or numpy arrays).
            config (Dict[str, Any]): Configuration dictionary, including perception parameters.
        """
        self.images = images # Can be paths or actual image data
        self.config = config
        self.cropper = ClassicalCVCropper(config) # Use the classical CV cropper
        self._solution_found = False
        self._solution = None
        
        # Initialize an empty list to store object IDs for the workspace
        # These are usually just indices for now, but could be more complex.
        self.objects = [f"obj_{i}" for i in range(len(images))] if images else []
        logger.info(f"SceneGraphBuilder initialized for {len(self.images)} images.")

    def extract_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
        """
        Extracts a specific feature (attribute or relation) for a given object.
        This method is primarily used by the emergent system's codelets.
        It relies on the underlying scene graph data.
        
        Args:
            obj_id (str): The ID of the object (e.g., "obj_0").
            feat_type (str): The type of feature to extract (e.g., "shape", "color", "left_of").
        Returns:
            Tuple[Any, float]: The extracted feature value and a confidence score.
        """
        # This method needs to access the *already built* scene graph for a specific image.
        # The `Workspace` will call this for a specific object within a specific image context.
        # For this to work, `SceneGraphBuilder` needs to store the scene graphs it builds.
        # Let's assume for simplicity that `obj_id` directly maps to an index in `self.objects`
        # and we are extracting features for the *first* image's scene graph.
        # In a multi-image context, `extract_feature` would need an `image_idx` argument.
        
        # For now, this method will be a placeholder that returns dummy values based on type,
        # as the full scene graph for a *specific image* isn't directly accessible here
        # without a context. The `build_scene_graph` method below is what actually
        # populates the scene graph.
        
        # If the emergent system needs to query features on *specific* objects from *specific* images,
        # the SceneGraphBuilder should be designed to hold pre-computed scene graphs,
        # or `extract_feature` should take an `image_idx` as an argument.
        
        # Given the current `main.py` setup, `extract_feature` is called in `Workspace`
        # with a generic `obj` (which is 'obj_X') and `feat`.
        # This means it's asking for a *general* feature, not necessarily from a specific image.
        # So, we return a mock value.
        
        # In a real system, the `Workspace` would pass the *current image's scene graph*
        # to `extract_feature` or `SceneGraphBuilder` would manage a cache of scene graphs.

        # For the purpose of making this non-dummy, we will return values that align
        # with the attributes/relations that `_get_scene_graph_from_masks` infers.
        # However, without a concrete image and object context, these are still general.

        # This part is tricky because `extract_feature` is called by codelets that don't
        # necessarily have the full image context. The `build_scene_graph` method is
        # what generates the full scene graph for an image.
        # For now, let's return a simple mock based on the feature type.
        # A more robust solution would involve the `Workspace` passing the current
        # image's scene graph to `extract_feature`.

        # This `extract_feature` is a placeholder for a *perception module*
        # that can extract features from a *specific object instance*.
        # It's not directly tied to the overall `build_scene_graph` process.
        
        # Given the current design, where `build_scene_graph` creates the full SG,
        # `extract_feature` should ideally query that SG.
        # Since `self.images` is just paths, we can't do real extraction here.
        # So, this remains a mock for now, but it's a mock that *could* be real
        # if `image_data` and `object_id` were passed directly.
        
        # The `Workspace`'s `Scout` codelet calls `sg.extract_feature(obj, feat)`.
        # This `obj` is like 'obj_0'. It's not tied to a specific image.
        # This suggests `extract_feature` should be able to get this from a *global*
        # understanding or a pre-computed knowledge base, or it needs the image context.

        # For the current setup, where `main.py` creates dummy images and passes them,
        # and `build_scene_graph` is called for *each* image, the `extract_feature`
        # here is slightly misaligned. It's designed for a low-level perception query.
        # Let's keep it as a mock that aligns with potential real output.
        
        # Example: if feat_type is 'shape', return 'circle'
        if feat_type == 'shape': return "circle", 0.8
        if feat_type == 'color': return "red", 0.7
        if feat_type == 'size': return "small", 0.6
        if feat_type == 'position_h': return "center_h", 0.5 # Simplified
        if feat_type == 'position_v': return "center_v", 0.5 # Simplified
        if feat_type == 'fill': return "solid", 0.7
        if feat_type == 'orientation': return "upright", 0.6
        if feat_type == 'texture': return "none", 0.6
        # For relations, it's more complex as it needs two objects.
        # This function is designed for single-object attribute extraction.
        # Relations are inferred in `_get_scene_graph_from_masks`.
        if feat_type in ["left_of", "right_of", "above", "below", "contains", "intersects", "aligned_horizontally", "aligned_vertically"]:
            return "true", 0.5 # Dummy for relations
        return "unknown", 0.1

    def problem_solved(self) -> bool:
        return self._solution_found
    
    def mark_solution(self, solution: Any):
        self._solution = solution
        self._solution_found = True

    def get_solution(self) -> Optional[Any]:
        return self._solution

    def build_scene_graph(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Constructs a scene graph for a single image.
        This method integrates the object detection (ClassicalCVCropper)
        and symbolic fusion logic.
        
        Args:
            image_np (np.ndarray): The input image as a NumPy array (H, W, 3).
        Returns:
            Dict[str, Any]: A dictionary representing the scene graph for the image.
                            Format: {'objects': [...], 'relations': [...], 'image_info': {...}}
        """
        if image_np is None or image_np.size == 0:
            logger.warning("No image provided for scene graph building.")
            return {'objects': [], 'relations': [], 'image_info': {}}

        logger.info(f"Building scene graph for image of shape {image_np.shape}.")
        
        # 1. Object Detection: Get masks using ClassicalCVCropper
        masks = self.cropper.detect_objects(image_np)
        
        if not masks:
            logger.info("No objects detected by cropper. Returning empty scene graph.")
            return {'objects': [], 'relations': [], 'image_info': {}}

        # 2. Extract basic geometric properties and initial attributes from masks
        objects_data = []
        for idx, mask in enumerate(masks):
            props = {
                'id': idx, # Assign an ID to each object
                'area': _mask_area(mask),
                'bbox_xyxy': _mask_to_bbox(mask), # [xmin, ymin, xmax, ymax]
                'centroid': _mask_centroid(mask),
                'aspect_ratio': _mask_aspect_ratio(mask),
                'solidity': _mask_solidity(mask),
                'mask': mask.tolist() # Store mask as list for JSON serialization (convert back to np.array if needed)
            }
            # For now, attributes are not directly extracted from raw pixels here.
            # They would come from a separate attribute classification model or a more advanced CV module.
            # We will generate dummy attributes for demonstration or assume they are passed.
            # For this integration, we'll assign dummy attributes if not provided.
            props['attributes'] = {
                'shape': random.choice(['circle', 'square', 'triangle']),
                'color': random.choice(['red', 'blue', 'green']),
                'size': random.choice(['small', 'medium', 'large'])
            }
            objects_data.append(props)
        logger.info(f"Extracted properties for {len(objects_data)} objects.")

        # 3. Build relation graph and infer spatial relations
        G = nx.Graph()
        for i, obj_i in enumerate(objects_data):
            G.add_node(i, **obj_i) # Add all object properties as node attributes

        relations_list = []
        image_height = image_np.shape[0]
        proximity_threshold = image_height * self.config.get('proximity_threshold_ratio', 0.1) # Configurable threshold

        for i in range(len(objects_data)):
            for j in range(i + 1, len(objects_data)):
                bbox_i = objects_data[i]['bbox_xyxy']
                bbox_j = objects_data[j]['bbox_xyxy']
                
                # Calculate Euclidean distance between centroids for proximity clustering
                dist = np.linalg.norm(np.array(objects_data[i]['centroid']) - np.array(objects_data[j]['centroid']))
                
                # Add specific spatial relation types as edge attributes and to relations_list
                spatial_rels = _get_spatial_relations(bbox_i, bbox_j, 
                                                      center_dist_thresh_px=self.config.get('center_dist_thresh_px', 10.0))
                
                for s_rel in spatial_rels:
                    # Add to NetworkX graph for clustering
                    G.add_edge(i, j, relation_type=s_rel, distance=float(dist)) # Store distance on edge
                    # Add to the relations list for the scene graph output
                    relations_list.append({
                        'subject_id': i,
                        'object_id': j,
                        'type': s_rel,
                        'confidence': 1.0 # Placeholder confidence
                    })
        logger.info(f"Inferred {len(relations_list)} spatial relations.")

        # 4. Grouping logic (e.g., cluster by proximity)
        clusters = _cluster_by_proximity(G, threshold=proximity_threshold)
        for idx, obj_data in enumerate(objects_data):
            obj_data['group_id'] = clusters.get(idx, -1) # Assign cluster ID
        logger.info(f"Performed object grouping. Found {len(set(clusters.values()))} groups.")

        # 5. Construct the final scene graph dictionary
        scene_graph = {
            'objects': objects_data,
            'relations': relations_list,
            'image_info': {
                'width': image_np.shape[1],
                'height': image_np.shape[0],
                'channels': image_np.shape[2]
            }
        }
        logger.info("Scene graph construction complete.")
        return scene_graph

