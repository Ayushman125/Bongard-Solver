# Folder: bongard_solver/

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import logging
import os

# Conditional imports for YOLO and SAM
try:
    from ultralytics import YOLO
    HAS_YOLO = True
    logger.info("Ultralytics YOLO found and enabled.")
except ImportError:
    logger.warning("Ultralytics YOLO not found. Object detection will be dummy.")
    HAS_YOLO = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
    logger.info("Segment Anything Model (SAM) found and enabled.")
except ImportError:
    logger.warning("Segment Anything Model (SAM) not found. Segmentation will be dummy.")
    HAS_SAM = False

logger = logging.getLogger(__name__)

# Global instances for models to avoid re-loading
_yolo_model: Optional[YOLO] = None
_sam_predictor: Optional[SamPredictor] = None

def load_yolo_and_sam_models(yolo_model_name: str, sam_model_type: str, sam_checkpoint_path: str, device: torch.device):
    """
    Loads the YOLO and SAM models. These are loaded once and stored globally.

    Args:
        yolo_model_name (str): Name of the YOLO model (e.g., 'yolov8n.pt').
        sam_model_type (str): Type of SAM model ('vit_h', 'vit_l', 'vit_b').
        sam_checkpoint_path (str): Path to the SAM checkpoint file.
        device (torch.device): Device to load models onto.
    """
    global _yolo_model, _sam_predictor

    if _yolo_model is None and HAS_YOLO:
        try:
            _yolo_model = YOLO(yolo_model_name)
            _yolo_model.to(device)
            logger.info(f"YOLO model '{yolo_model_name}' loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{yolo_model_name}': {e}")
            _yolo_model = None

    if _sam_predictor is None and HAS_SAM:
        try:
            if not os.path.exists(sam_checkpoint_path):
                logger.error(f"SAM checkpoint not found at {sam_checkpoint_path}. Please download it.")
                _sam_predictor = None
                return

            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
            sam.to(device)
            _sam_predictor = SamPredictor(sam)
            logger.info(f"SAM model '{sam_model_type}' loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load SAM model '{sam_model_type}' from '{sam_checkpoint_path}': {e}")
            _sam_predictor = None

def detect_and_segment_image(
    image_np: np.ndarray, # H, W, C (RGB)
    yolo_conf_threshold: float,
    yolo_iou_threshold: float,
    max_objects: int
) -> Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Performs object detection using YOLO and refines with SAM segmentation for a single image.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (H, W, C) in RGB format.
        yolo_conf_threshold (float): Confidence threshold for YOLO detections.
        yolo_iou_threshold (float): IoU threshold for NMS in YOLO.
        max_objects (int): Maximum number of objects to return.

    Returns:
        Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
            - bboxes (List[List[float]]): List of refined bounding boxes (x1, y1, x2, y2).
            - masks (List[np.ndarray]): List of binary masks (H, W) for each detected object.
            - scene_graph_objects (List[Dict[str, Any]]): List of object dictionaries for scene graph.
    """
    if _yolo_model is None or _sam_predictor is None:
        logger.warning("YOLO or SAM models not loaded. Returning dummy detection and segmentation.")
        h, w, _ = image_np.shape
        dummy_bboxes = []
        dummy_masks = []
        dummy_scene_graph_objects = []
        num_dummy_objects = random.randint(1, max_objects)
        for i in range(num_dummy_objects):
            x1 = random.uniform(0, w * 0.8)
            y1 = random.uniform(0, h * 0.8)
            x2 = random.uniform(x1 + 10, w)
            y2 = random.uniform(y1 + 10, h)
            bbox = [x1, y1, x2, y2]
            mask = np.zeros((h, w), dtype=bool)
            cv2.rectangle(mask.astype(np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), 1, -1) # Fill dummy mask
            
            dummy_bboxes.append(bbox)
            dummy_masks.append(mask)
            dummy_scene_graph_objects.append({
                'id': i,
                'bbox': bbox,
                'mask': mask, # Store mask directly in scene graph for now
                'attributes': {},
                'relations': []
            })
        return dummy_bboxes, dummy_masks, dummy_scene_graph_objects

    # 1. YOLO Detection
    yolo_results = _yolo_model(image_np, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False)
    
    boxes_xyxy = []
    if yolo_results and len(yolo_results) > 0:
        # Extract boxes from the first result (assuming batch size 1 for single image)
        boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes_xyxy) == 0:
        logger.debug("No objects detected by YOLO.")
        return [], [], []

    # Limit to max_objects
    if len(boxes_xyxy) > max_objects:
        # Sort by confidence and take top max_objects
        confidences = yolo_results[0].boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(confidences)[::-1]
        boxes_xyxy = boxes_xyxy[sorted_indices[:max_objects]]
        logger.debug(f"Limited YOLO detections to top {max_objects} objects.")
    
    # 2. SAM Segmentation
    _sam_predictor.set_image(image_np)
    
    masks_list = []
    refined_bboxes_list = []
    scene_graph_objects = []

    for i, box in enumerate(boxes_xyxy):
        input_box = np.array(box)
        
        # SAM prediction
        mask, _, _ = _sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :], # SAM expects (1, 4) box
            multimask_output=False # Get a single mask
        )
        mask = mask[0] # mask is (1, H, W) bool, get (H, W) bool
        masks_list.append(mask)

        # Refine bounding box from mask
        # Find contours of the mask to get a tighter bounding box
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour (assuming one main object per mask)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            refined_bbox = [float(x), float(y), float(x + w), float(y + h)]
            refined_bboxes_list.append(refined_bbox)
        else:
            refined_bboxes_list.append(box.tolist()) # Fallback to YOLO box if no mask contour

        # Prepare object for scene graph
        scene_graph_objects.append({
            'id': i,
            'bbox': refined_bboxes_list[-1], # Use refined bbox
            'mask': mask, # Store the boolean mask
            'attributes': {}, # To be filled by AttributeClassifier
            'relations': [] # To be filled by RelationGNN
        })

    logger.debug(f"Detected and segmented {len(masks_list)} objects.")
    return refined_bboxes_list, masks_list, scene_graph_objects


def get_masked_crop(image_tensor: torch.Tensor, bbox: List[float], mask_np: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Extracts a masked crop from an image tensor, resizes it, and returns it.

    Args:
        image_tensor (torch.Tensor): The original image tensor (C, H, W).
        bbox (List[float]): Bounding box [x1, y1, x2, y2].
        mask_np (np.ndarray): Binary mask (H, W) for the object.
        target_size (Tuple[int, int]): Desired output size (H_out, W_out).

    Returns:
        torch.Tensor: The masked and resized crop (C, H_out, W_out).
    """
    _, img_h, img_w = image_tensor.shape
    x1, y1, x2, y2 = [int(b) for b in bbox]

    # Clamp bbox coordinates to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bounding box for cropping: {bbox}. Returning black image.")
        return torch.zeros(image_tensor.shape[0], *target_size, device=image_tensor.device)

    # Crop the image using the bounding box
    cropped_image = image_tensor[:, y1:y2, x1:x2] # (C, crop_H, crop_W)
    cropped_mask = torch.from_numpy(mask_np[y1:y2, x1:x2]).to(image_tensor.device).unsqueeze(0).float() # (1, crop_H, crop_W)

    # Resize cropped image and mask to target_size
    # Use bilinear for image, nearest for mask to keep it binary
    resized_cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    resized_cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)

    # Apply mask to the resized crop
    # Expand mask to C channels
    masked_crop = resized_cropped_image * resized_cropped_mask

    return masked_crop


def mask_to_yolo_format(mask: np.ndarray, image_shape: Tuple[int, int]) -> Optional[str]:
    """
    Converts a binary mask to YOLO bounding box format (class_id cx cy nw nh).
    Assumes class_id 0 for all objects.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        image_shape (Tuple[int, int]): Original image shape (H, W).

    Returns:
        Optional[str]: YOLO formatted string or None if no valid contour.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Convert to YOLO format (normalized center_x, center_y, width, height)
    img_h, img_w = image_shape
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

