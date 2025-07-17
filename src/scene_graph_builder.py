# Folder: bongard_solver/src/
# File: scene_graph_builder.py
import numpy as np
import networkx as nx
import logging
import cv2 # For mask processing and contour detection
from PIL import Image # For opening images
import math # For geometric calculations
from collections import defaultdict
import json # For saving/loading symbolic annotations
from typing import List, Dict, Any, Tuple, Optional
import torch # For SAM and CNN features

# Configure logging for this module
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import CONFIG
except ImportError:
    logger.error("Could not import CONFIG from config.py. SceneGraphBuilder will use default values.")
    CONFIG = {'model': {},
              'segmentation': {'use_sam': False, 'sam_model_type': 'vit_b', 'sam_checkpoint_path': '', 'sam_points_per_side': 32, 'sam_pred_iou_thresh': 0.88},
              'debug': {'max_fallback_cnt': 5, 'min_contour_area_sam_fallback': 50},
              'use_cnn_features': True, # Default to True for primitive_extractor
              'proximity_threshold_ratio': 0.1, # For spatial relations
              'center_dist_thresh_px': 10.0 # For spatial relations
             },
             'data': {'image_size': [224, 224]} # Default image size
            }

# Import primitive_extractor for attribute extraction with confidence
try:
    from src.perception.primitive_extractor import extract_shape_conf, extract_fill_conf, extract_cnn_features
    HAS_PRIMITIVE_EXTRACTOR = True
except ImportError:
    logger.warning("Could not import primitive_extractor.py. SceneGraphBuilder will use dummy attribute extraction.")
    HAS_PRIMITIVE_EXTRACTOR = False
    def extract_shape_conf(img): return "dummy_shape", 0.5
    def extract_fill_conf(img): return "dummy_fill", 0.5
    def extract_cnn_features(img): return {"shape": ("dummy_cnn_shape", 0.6), "color": ("dummy_cnn_color", 0.7), "size": ("dummy_cnn_size", 0.5), "fill": ("dummy_cnn_fill", 0.6), "orientation": ("dummy_cnn_orientation", 0.5), "texture": ("dummy_cnn_texture", 0.5)}

# Import SAM for segmentation
HAS_SAM_SEG = False
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    HAS_SAM_SEG = True
    logger.info("Segment Anything Model (SAM) found and enabled for SceneGraphBuilder.")
except ImportError:
    logger.warning("Segment Anything Model (SAM) not found. SAM segmentation will be disabled in SceneGraphBuilder.")



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
        float: Aspect ratio, or 0.0 if mask or height is zero.
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
        relations.append('left_of')
    elif cx1 > cx2 + center_dist_thresh_px:
        relations.append('right_of')
    
    # Vertical relations (smaller y is higher)
    if cy1 < cy2 - center_dist_thresh_px:
        relations.append('above')
    elif cy1 > cy2 + center_dist_thresh_px:
        relations.append('below')

    # Overlap
    current_iou = iou_xyxy(bbox1, bbox2)
    if current_iou > threshold_iou:
        relations.append('intersects')
    
    # Contains (A contains B)
    if (bbox1[0] <= bbox2[0] and bbox1[1] <= bbox2[1] and
        bbox1[2] >= bbox2[2] and bbox1[3] >= bbox2[3]):
        relations.append('contains')
    
    # Is contained by (A is contained by B)
    if (bbox2[0] <= bbox1[0] and bbox2[1] <= bbox1[1] and
        bbox2[2] >= bbox1[2] and bbox2[3] >= bbox1[3]):
        relations.append('is_contained_by')
    
    # Alignment (horizontal or vertical)
    if abs(cx1 - cx2) < center_dist_thresh_px:
        relations.append('aligned_vertically')
    if abs(cy1 - cy2) < center_dist_thresh_px:
        relations.append('aligned_horizontally')

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

# --- Object Detection/Segmentation ---
class ObjectDetector:
    """
    Handles object detection using SAM (if available) or a classical CV fallback.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_sam = config['segmentation']['use_sam'] and HAS_SAM_SEG
        self.sam_predictor = None
        self.sam_mask_generator = None
        if self.use_sam:
            try:
                sam_checkpoint = self.config['segmentation']['sam_checkpoint_path']
                model_type = self.config['segmentation']['sam_model_type']
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                self.sam_predictor = SamPredictor(sam)
                self.sam_mask_generator = SamAutomaticMaskGenerator(
                    sam, 
                    points_per_side=self.config['segmentation']['sam_points_per_side'],
                    pred_iou_thresh=self.config['segmentation']['sam_pred_iou_thresh']
                )
                logger.info(f"SAM {model_type} loaded for segmentation.")
            except Exception as e:
                logger.error(f"Failed to load SAM model: {e}. Disabling SAM segmentation.")
                self.use_sam = False
        # Classical CV fallback parameters
        self.min_contour_area_ratio = self.config['debug'].get('min_contour_area_sam_fallback', 50) / (CONFIG['data']['image_size'][0] * CONFIG['data']['image_size'][1]) # Convert to ratio
        self.max_fallback_cnt = self.config['debug'].get('max_fallback_cnt', 5)
        logger.info(f"ObjectDetector initialized. Use SAM: {self.use_sam}")

    def detect_and_segment(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects objects and generates masks. Prioritizes SAM, then CV fallback.
        Returns a list of dictionaries, each with 'bbox', 'mask', 'label', 'confidence'.
        """
        if image_np is None or image_np.size == 0:
            logger.warning("Input image is empty or None for object detection.")
            return []
        detections = []
        if self.use_sam:
            logger.debug("Attempting SAM automatic mask generation.")
            try:
                sam_results = self.sam_mask_generator.generate(image_np)
                for mask_data in sam_results:
                    mask = mask_data['segmentation'].astype(np.uint8) * 255
                    bbox = mask_data['bbox'] # xywh format
                    bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                    detections.append({
                        'bbox': bbox_xyxy,
                        'mask': mask,
                        'label': 'object',
                        'confidence': mask_data.get('stability_score', 1.0)
                    })
                logger.info(f"SAM detected {len(detections)} objects.")
                if detections: return detections
            except Exception as e:
                logger.error(f"Error during SAM mask generation: {e}. Falling back to CV.", exc_info=True)
                self.use_sam = False
        logger.info("Falling back to Classical CV (contour detection) for objects.")
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_fallback_cnt]
        image_area = image_np.shape[0] * image_np.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image_area * self.min_contour_area_ratio:
                mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                bbox_xyxy = cv2.boundingRect(contour) # xywh
                bbox_xyxy = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[0]+bbox_xyxy[2], bbox_xyxy[1]+bbox_xyxy[3]]
                detections.append({
                    'bbox': bbox_xyxy,
                    'mask': (mask > 0).astype(np.uint8) * 255,
                    'label': 'object_cv',
                    'confidence': area / image_area
                })
        logger.info(f"Classical CV detected {len(detections)} objects.")
        return detections

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
        self.object_detector = ObjectDetector(config) # Use the unified ObjectDetector
        self._solution_found = False
        self._solution = None
        
        # Initialize an empty list to store object IDs for the workspace
        # These are usually just indices for now, but could be more complex.
        self.objects = [f"obj_{i}" for i in range(len(images))] if images else []
        
        # Cache for object images (crops) to avoid re-cropping for attribute extraction
        self._object_image_cache: Dict[str, Image.Image] = {} 
        logger.info(f"SceneGraphBuilder initialized for {len(self.images)} images.")

    def get_object_image(self, obj_id: str) -> Optional[Image.Image]:
        """
        Retrieves a PIL Image crop for a given object ID from the cache.
        This method assumes that `build_scene_graph` has already been called
        and populated the cache.
        """
        return self._object_image_cache.get(obj_id)

    def extract_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
        """
        Extracts a specific feature (attribute or relation) for a given object.
        This method is primarily used by the emergent system's codelets.
        It relies on the underlying scene graph data or direct primitive extraction.
        
        Args:
            obj_id (str): The ID of the object (e.g., "obj_0").
            feat_type (str): The type of feature to extract (e.g., "shape", "color", "left_of").
        Returns:
            Tuple[Any, float]: The extracted feature value and a confidence score.
        """
        obj_image = self.get_object_image(obj_id)
        if obj_image is None:
            logger.warning(f"No image available in cache for object {obj_id}. Cannot extract feature {feat_type}. Returning dummy.")
            # Fallback to dummy extraction if no image is available
            return "unknown", 0.1
        
        val, conf = "unknown", 0.0
        if HAS_PRIMITIVE_EXTRACTOR:
            # Use CNN-based features if configured, otherwise classical CV
            if self.config['model'].get('use_cnn_features', True): # Default to CNN features
                cnn_feats = extract_cnn_features(obj_image)
                if feat_type in cnn_feats:
                    val, conf = cnn_feats[feat_type]
                else:
                    logger.warning(f"CNN features for {feat_type} not available. Falling back to CV.")
                    if feat_type == 'shape': val, conf = extract_shape_conf(obj_image)
                    elif feat_type == 'fill': val, conf = extract_fill_conf(obj_image)
                    else: val, conf = "unknown", 0.1 # Default for unsupported CV features
            else: # Use classical CV features
                if feat_type == 'shape': val, conf = extract_shape_conf(obj_image)
                elif feat_type == 'fill': val, conf = extract_fill_conf(obj_image)
                # Add more CV-based feature extractions as needed
                else:
                    logger.warning(f"Unsupported feature type '{feat_type}' for classical CV. Using dummy.")
                    val, conf = "unknown", 0.1
        else:
            # Fallback to simple mock if primitive_extractor is not available
            if feat_type == 'shape': val, conf = random.choice(['circle', 'square', 'triangle']), random.uniform(0.7, 0.9)
            elif feat_type == 'color': val, conf = random.choice(['red', 'blue', 'green']), random.uniform(0.6, 0.8)
            elif feat_type == 'size': val, conf = random.choice(['small', 'medium', 'large']), random.uniform(0.7, 0.9)
            elif feat_type == 'fill': val, conf = random.choice(['filled', 'outlined']), random.uniform(0.6, 0.8)
            elif feat_type == 'orientation': val, conf = random.choice(['horizontal', 'vertical']), random.uniform(0.6, 0.8)
            elif feat_type == 'texture': val, conf = random.choice(['smooth', 'rough']), random.uniform(0.6, 0.8)
            else: val, conf = "unknown", 0.1 # Default for relations or unknown types

        logger.debug(f"Extracted feature for {obj_id}, {feat_type}: {val} (conf: {conf:.4f})")
        return val, conf

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
        This method integrates the object detection (ObjectDetector)
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
        
        # 1. Object Detection and Segmentation
        detected_objects = self.object_detector.detect_and_segment(image_np)
        
        if not detected_objects:
            logger.info("No objects detected. Returning empty scene graph.")
            return {'objects': [], 'relations': [], 'image_info': {}}

        # 2. Extract basic geometric properties and initial attributes from masks/bboxes
        objects_data = []
        self._object_image_cache.clear() # Clear cache for new image
        for idx, obj_det in enumerate(detected_objects):
            mask = obj_det['mask']
            bbox_xyxy = obj_det['bbox'] # Already in xyxy format
            
            # Crop object image for attribute extraction
            xmin, ymin, xmax, ymax = bbox_xyxy
            obj_image_np = image_np[ymin:ymax, xmin:xmax]
            obj_image_pil = Image.fromarray(obj_image_np)
            
            # Store cropped image in cache for later `extract_feature` calls
            obj_id_str = f"obj_{idx}"
            self._object_image_cache[obj_id_str] = obj_image_pil
            
            # Calculate geometric properties
            area = _mask_area(mask) if mask is not None else (bbox_xyxy[2]-bbox_xyxy[0])*(bbox_xyxy[3]-bbox_xyxy[1])
            centroid = _mask_centroid(mask) if mask is not None else [(bbox_xyxy[0]+bbox_xyxy[2])/2, (bbox_xyxy[1]+bbox_xyxy[3])/2]
            aspect_ratio = _mask_aspect_ratio(mask) if mask is not None else (bbox_xyxy[2]-bbox_xyxy[0])/(bbox_xyxy[3]-bbox_xyxy[1] + 1e-6)
            solidity = _mask_solidity(mask) if mask is not None else 1.0 # Bbox solidity is 1.0

            props = {
                'id': obj_id_str, # Assign an ID to each object
                'area': area,
                'bbox_xyxy': bbox_xyxy, # [xmin, ymin, xmax, ymax]
                'centroid': centroid,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'mask': mask.tolist() if mask is not None else [] # Store mask as list for JSON serialization
            }
            
            # Extract attributes using primitive_extractor with confidence
            attributes = {}
            attribute_confidences = {}
            for feat_type in ['shape', 'color', 'size', 'fill', 'orientation', 'texture']:
                val, conf = self.extract_feature(obj_id_str, feat_type) # Use the unified extract_feature
                attributes[feat_type] = val
                attribute_confidences[feat_type] = conf
            
            props['attributes'] = attributes
            props['attribute_confidences'] = attribute_confidences # Store confidences
            props['detection_confidence'] = obj_det.get('confidence', 1.0) # Confidence from detector
            props['label'] = obj_det.get('label', 'object') # Label from detector
            
            objects_data.append(props)
        logger.info(f"Extracted properties for {len(objects_data)} objects.")

        # 3. Build relation graph and infer spatial relations
        G = nx.Graph()
        for i, obj_i in enumerate(objects_data):
            G.add_node(i, **obj_i) # Add all object properties as node attributes
        
        relations_list = []
        image_height = image_np.shape[0]
        image_width = image_np.shape[1]
        
        proximity_threshold = image_height * self.config['model'].get('proximity_threshold_ratio', 0.1) # Configurable threshold
        center_dist_thresh_px = self.config['model'].get('center_dist_thresh_px', 10.0)

        for i in range(len(objects_data)):
            for j in range(i + 1, len(objects_data)):
                bbox_i = objects_data[i]['bbox_xyxy']
                bbox_j = objects_data[j]['bbox_xyxy']
                
                # Calculate Euclidean distance between centroids for proximity clustering
                dist = np.linalg.norm(np.array(objects_data[i]['centroid']) - np.array(objects_data[j]['centroid']))
                
                # Add specific spatial relation types as edge attributes and to relations_list
                spatial_rels = _get_spatial_relations(bbox_i, bbox_j, 
                                                      threshold_iou=self.config['model']['yolo_iou_threshold'], # Use YOLO IoU threshold
                                                      center_dist_thresh_px=center_dist_thresh_px)
                
                for s_rel in spatial_rels:
                    # Add to NetworkX graph for clustering
                    G.add_edge(i, j, relation_type=s_rel, distance=float(dist)) # Store distance on edge
                    # Add to the relations list for the scene graph output
                    relations_list.append({
                        'subject_id': objects_data[i]['id'], # Use string IDs
                        'object_id': objects_data[j]['id'], # Use string IDs
                        'type': s_rel,
                        'confidence': 1.0 # Placeholder confidence, could be learned
                    })
                    # Also add inverse relation
                    # Note: You might want to define explicit inverse types (e.g., 'right_of' for 'left_of')
                    # For simplicity here, just appending '_inverse'
                    relations_list.append({
                        'subject_id': objects_data[j]['id'],
                        'object_id': objects_data[i]['id'],
                        'type': s_rel + '_inverse', # e.g., 'right_of_inverse' for 'left_of'
                        'confidence': 1.0
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
