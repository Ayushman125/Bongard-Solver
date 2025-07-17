import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F # Added for F.interpolate

logger = logging.getLogger(__name__)

# Try to import SAM
HAS_SAM = False
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor # Added SamPredictor for consistency
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    logger.warning("Segment Anything Model (SAM) not found. Only classical CV fallback will be used.")

# Import configuration (if needed by this file, otherwise remove)
try:
    from config import CONFIG
except ImportError:
    logger.warning("Could not import CONFIG from config.py. Using default values for SAM/CV parameters.")
    CONFIG = {
        'segmentation': {'sam_model_type': 'vit_b', 'sam_checkpoint_path': ''},
        'debug': {'min_contour_area_sam_fallback': 50} # Placeholder if needed
    }


def detect_and_segment_image(image_np: np.ndarray, max_objects: int = 10) -> Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Detects and segments objects in an image using SAM (if available) or classical CV fallback.

    Args:
        image_np (np.ndarray): Input image (H, W, 3) RGB.
        max_objects (int): Max number of objects to return.

    Returns:
        Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
            - bboxes: List of [x1, y1, x2, y2]
            - masks: List of binary masks (H, W)
            - scene_graph_objects: List of dicts for scene graph
    """
    bboxes = []
    masks = []
    scene_graph_objects = []

    h, w, _ = image_np.shape # Get image dimensions for later use

    if HAS_SAM:
        try:
            # Use default model type and checkpoint path from config if available
            model_type = CONFIG.get('segmentation', {}).get('sam_model_type', 'vit_b')
            checkpoint_path = CONFIG.get('segmentation', {}).get('sam_checkpoint_path', '')

            if not checkpoint_path or not os.path.exists(checkpoint_path):
                logger.warning(f"SAM checkpoint not found at {checkpoint_path}. Falling back to CV.")
                raise RuntimeError("SAM checkpoint missing or invalid path")

            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            # Ensure device handling is correct for torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device)

            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=CONFIG['segmentation'].get('sam_points_per_side', 32), # Use config value or default
                pred_iou_thresh=CONFIG['segmentation'].get('sam_pred_iou_thresh', 0.88) # Use config value or default
            )
            sam_results = mask_generator.generate(image_np)

            for mask_data in sam_results[:max_objects]:
                mask = mask_data['segmentation'].astype(np.uint8) * 255
                bbox = mask_data['bbox']  # xywh
                bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                bboxes.append(bbox_xyxy)
                masks.append(mask)
                scene_graph_objects.append({
                    'bbox': bbox_xyxy,
                    'mask': mask.tolist(),
                    'label': 'object',
                    'confidence': mask_data.get('stability_score', 1.0)
                })
            logger.info(f"SAM detected {len(bboxes)} objects.")
            return bboxes, masks, scene_graph_objects
        except Exception as e:
            logger.error(f"SAM failed: {e}. Falling back to classical CV.", exc_info=True) # Added exc_info

    # Classical CV fallback
    logger.info("Using classical CV fallback for object detection.")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]
    image_area = image_np.shape[0] * image_np.shape[1]

    for i, contour in enumerate(contours): # Added 'i' for object ID consistency
        area = cv2.contourArea(contour)
        # Use a configurable min_contour_area_sam_fallback from CONFIG
        min_area_threshold = CONFIG['debug'].get('min_contour_area_sam_fallback', 50)
        if area > min_area_threshold:  # Filter out tiny contours based on config
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8) # Use image_np.shape[:2] for mask dimensions
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            x, y, w_bbox, h_bbox = cv2.boundingRect(contour) # Renamed w, h to w_bbox, h_bbox to avoid conflict with image h, w
            bbox_xyxy = [x, y, x+w_bbox, y+h_bbox]
            bboxes.append(bbox_xyxy)
            masks.append(mask)
            scene_graph_objects.append({
                'bbox': bbox_xyxy,
                'mask': mask.tolist(), # Convert mask to list for JSON serialization
                'label': 'object_cv',
                'confidence': area / image_area
            })
    logger.info(f"Classical CV detected {len(bboxes)} objects.")
    return bboxes, masks, scene_graph_objects


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
    cropped_image = image_tensor[:, y1:y2, x1:x2]  # (C, crop_H, crop_W)
    cropped_mask = torch.from_numpy(mask_np[y1:y2, x1:x2]).to(image_tensor.device).unsqueeze(0).float()  # (1, crop_H, crop_W)
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
