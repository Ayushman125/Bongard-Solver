from skimage.measure import label
from .skeleton import process_and_analyze_skeleton
from .shape_utils import estimate_stroke_width
from .persistence import compute_betti
from .structure import structure_tensor_coherence

def extract_shapes_and_metrics(mask, area_thresh=300, aspect_thresh=0.5, invert_if_needed=True):
    """
    Extracts all foreground shapes from a mask, computes per-shape metrics, and returns results.

    Args:
        mask (np.ndarray): Input mask (0/255 or 0/1, uint8 or bool).
        area_thresh (int): Minimum area for a shape to be considered.
        aspect_thresh (float): Minimum aspect ratio for a bounding box.
        invert_if_needed (bool): If True, will invert mask if background is larger than any shape.

    Returns:
        List[dict]: Each dict contains 'mask', 'bbox', and metrics for a shape.
    """
    # Ensure mask is 0/1 uint8
    mask = (mask > 0).astype(np.uint8)

    # Optionally invert mask if background is largest
    if invert_if_needed:
        # Use connected components to find largest area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            bg_area = stats[0, cv2.CC_STAT_AREA]
            max_fg_area = stats[1:, cv2.CC_STAT_AREA].max() if num_labels > 2 else 0
            if bg_area > max_fg_area:
                mask = 1 - mask

    # Connected components (ignore background label 0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    results = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < area_thresh:
            continue
        x, y, w, h = stats[label_idx, cv2.CC_STAT_LEFT], stats[label_idx, cv2.CC_STAT_TOP], stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect_ratio < aspect_thresh:
            continue
        # Per-shape mask
        shape_mask = (labels[y:y+h, x:x+w] == label_idx).astype(np.uint8)
        # Metrics
        skel_img, skel_metrics = process_and_analyze_skeleton(shape_mask)
        stroke_width = estimate_stroke_width(shape_mask)
        betti0, betti1 = compute_betti(shape_mask)
        coherence = structure_tensor_coherence(shape_mask)
        results.append({
            'mask': shape_mask,
            'bbox': (x, y, w, h),
            'skeleton': skel_img,
            'skeleton_metrics': skel_metrics,
            'stroke_width': stroke_width,
            'betti0': betti0,
            'betti1': betti1,
            'coherence': coherence
        })
    return results
import cv2
import numpy as np

def find_quadrangles(mask, area_thresh=300, aspect_thresh=0.5):
    """
    Finds quadrangles in a binary mask.

    Args:
        mask (np.ndarray): Input binary mask (0s and 255s).
        area_thresh (int): Minimum contour area to be considered a shape.
        aspect_thresh (float): Minimum aspect ratio for a bounding box.

    Returns:
        list: A list of contours, where each contour is a list of points.
    """
    # Ensure the mask is a binary, 8-bit, single-channel image.
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Using a simple binary threshold as the input should be clean.
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue

        # Check aspect ratio to filter out line-like shapes
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect_ratio < aspect_thresh:
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            quadrangles.append(approx)
    return quadrangles
