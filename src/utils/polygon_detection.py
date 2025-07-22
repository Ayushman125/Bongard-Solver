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
