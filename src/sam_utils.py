
# Conditional imports for SAM
import os

# Conditional imports for YOLO and SAM
try:
    from ultralytics import YOLO
    HAS_YOLO = True
    logger.info("Ultralytics YOLO found and enabled.")
except ImportError:
    logger.warning("Ultralytics YOLO not found. Object detection will be dummy.")
    HAS_YOLO = False
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
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

        # Prepare object for scene graph
    image_np: np.ndarray,
    max_objects: int = 10
) -> Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Performs object detection and segmentation using SAM (if available) or classical CV fallback.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (H, W, C) in RGB format.
        max_objects (int): Maximum number of objects to return.

    Returns:
        Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
            - bboxes (List[List[float]]): List of bounding boxes (x1, y1, x2, y2).
            - masks (List[np.ndarray]): List of binary masks (H, W) for each detected object.
            - scene_graph_objects (List[Dict[str, Any]]): List of object dictionaries for scene graph.
    """
    logger = logging.getLogger(__name__)
    h, w, _ = image_np.shape
    bboxes = []
    masks = []
    scene_graph_objects = []

    if _sam_predictor is not None:
        try:
            _sam_predictor.set_image(image_np)
            # Use a grid of points or a single box covering the image for demo; real use may differ
            # Here, we use a single box covering the whole image as a fallback
            input_box = np.array([0, 0, w, h])
            mask, _, _ = _sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            mask = mask[0]
            # Find contours to get bounding box
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                bbox = [float(x), float(y), float(x + bw), float(y + bh)]
            else:
                bbox = [0.0, 0.0, float(w), float(h)]
            bboxes.append(bbox)
            masks.append(mask)
            scene_graph_objects.append({
                'id': 0,
                'bbox': bbox,
                'mask': mask,
                'attributes': {},
                'relations': []
            })
            logger.info(f"SAM detected 1 object.")
            return bboxes, masks, scene_graph_objects
        except Exception as e:
            logger.error(f"Error during SAM mask generation: {e}. Falling back to classical CV.")

    # Classical CV fallback
    logger.info("Falling back to Classical CV (contour detection) for objects.")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(x + bw), float(y + bh)]
            bboxes.append(bbox)
            masks.append(mask)
            scene_graph_objects.append({
                'id': i,
                'bbox': bbox,
                'mask': mask,
                'attributes': {},
                'relations': []
            })
    logger.info(f"Classical CV detected {len(bboxes)} objects.")
    return bboxes, masks, scene_graph_objects
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

