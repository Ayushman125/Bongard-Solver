# Folder: bongard_solver/src/
# File: fast_props.py
import cv2
import numpy as np
from skimage.measure import label, regionprops
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_props(gray_img: np.ndarray, min_area: float, max_cnt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts object properties (bounding boxes, areas) from a grayscale image
    using scikit-image's C/Cython-optimized routines.
    Args:
        gray_img (np.ndarray): The input grayscale image.
        min_area (float): Minimum contour area to consider.
        max_cnt (int): Maximum number of contours to extract.
    Returns:
        tuple: A tuple containing:
            - boxes (np.ndarray): Bounding boxes in (x, y, w, h) format, shape (N, 4).
            - areas (np.ndarray): Areas of the detected objects, shape (N,).
    """
    if gray_img is None or gray_img.size == 0:
        logger.warning("Received empty or None image for extract_props. Returning empty arrays.")
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Ensure block_size is odd for adaptiveThreshold
    block_size = 11 # Example value, can be passed from config if needed
    C_val = 2     # Example value, can be passed from config if needed

    # 1) Binary threshold using adaptive thresholding
    # Convert to uint8 if not already
    if gray_img.dtype != np.uint8:
        gray_img = (gray_img / gray_img.max() * 255).astype(np.uint8)
    
    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size <= 1:
        block_size = 3 # Minimum valid block size

    bw = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C_val
    )

    # 2) Label connected components (skimage.measure.label)
    lbl = label(bw, connectivity=2) # connectivity=2 for 8-connectivity
    props = regionprops(lbl, intensity_image=gray_img)

    # 3) Extract numeric arrays and filter
    boxes: List[List[float]] = []
    areas: List[float] = []
    
    # Sort by area in descending order and then filter
    sorted_props = sorted(props, key=lambda p: p.area, reverse=True)
    for prop in sorted_props:
        if prop.area < min_area:
            continue
        
        # prop.bbox returns (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = prop.bbox
        w, h = maxc - minc, maxr - minr
        
        # Ensure positive dimensions
        if w <= 0 or h <= 0:
            continue
        
        # Store as (x, y, w, h) for consistency with YOLO format
        boxes.append([float(minc), float(minr), float(w), float(h)])
        areas.append(float(prop.area))
        
        if len(boxes) >= max_cnt:
            break
            
    # Convert to numpy arrays
    boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
    areas_np = np.array(areas, dtype=np.float32) if areas else np.empty((0,), dtype=np.float32)
    
    return boxes_np, areas_np
