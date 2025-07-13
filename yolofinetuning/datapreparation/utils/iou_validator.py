import numpy as np

def compute_iou(boxA, boxB):
    # Standard IoU implementation
    xA1, yA1, wA, hA = boxA['x'], boxA['y'], boxA['w'], boxA['h']
    xB1, yB1, wB, hB = boxB['x'], boxB['y'], boxB['w'], boxB['h']
    def to_corners(x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    a1, a2, a3, a4 = to_corners(xA1, yA1, wA, hA)
    b1, b2, b3, b4 = to_corners(xB1, yB1, wB, hB)
    xi1, yi1 = max(a1, b1), max(a2, b2)
    xi2, yi2 = min(a3, b3), min(a4, b4)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    areaA = (a3 - a1) * (a4 - a2)
    areaB = (b3 - b1) * (b4 - b2)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0

def find_low_iou_pairs(ground_truth, auto_labels, threshold=0.5):
    flags = []
    for gt in ground_truth:
        for al in auto_labels:
            if compute_iou(gt['bbox'], al['bbox']) < threshold:
                flags.append((gt, al))
    return flags
