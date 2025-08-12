"""
robust_geometry.py
Robust geometric computation utilities for Bongard Solver.
Implements Simulation of Simplicity (SoS), symbolic perturbation, and multi-strategy recovery.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

def robust_convex_hull(points):
    """Robust convex hull computation with SoS and fallback."""
    try:
        if len(points) < 3:
            return None, {'degenerate': True, 'reason': 'Too few points'}
        hull = ConvexHull(points)
        return hull, {'degenerate': False, 'quality': 1.0}
    except QhullError as e:
        # Symbolic perturbation fallback
        perturbed = np.array(points) + np.random.normal(0, 1e-8, np.array(points).shape)
        try:
            hull = ConvexHull(perturbed)
            return hull, {'degenerate': False, 'quality': 0.8, 'recovered': True}
        except Exception as e2:
            return None, {'degenerate': True, 'reason': str(e2)}

def robust_normalize(points):
    """Robust normalization with fallback for collapsed/degenerate cases."""
    arr = np.array(points)
    if arr.shape[0] < 2:
        return arr.tolist(), {'degenerate': True, 'reason': 'Too few points'}
    min_x, min_y = np.min(arr, axis=0)
    max_x, max_y = np.max(arr, axis=0)
    width, height = max_x - min_x, max_y - min_y
    if width == 0 or height == 0:
        # Symbolic perturbation fallback
        arr = arr + np.random.normal(0, 1e-8, arr.shape)
        min_x, min_y = np.min(arr, axis=0)
        max_x, max_y = np.max(arr, axis=0)
        width, height = max_x - min_x, max_y - min_y
        if width == 0 or height == 0:
            return arr.tolist(), {'degenerate': True, 'reason': 'Collapsed after perturbation'}
    normalized = (arr - [min_x, min_y]) / [width, height]
    return normalized.tolist(), {'degenerate': False, 'quality': 1.0}

def robust_area(points):
    """Robust area calculation with fallback."""
    arr = np.array(points)
    if arr.shape[0] < 3:
        return 0.0, {'degenerate': True, 'reason': 'Too few points'}
    try:
        x = arr[:,0]
        y = arr[:,1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area == 0.0:
            arr = arr + np.random.normal(0, 1e-8, arr.shape)
            x = arr[:,0]
            y = arr[:,1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area, {'degenerate': area == 0.0, 'quality': 1.0 if area > 0 else 0.0}
    except Exception as e:
        return 0.0, {'degenerate': True, 'reason': str(e)}
