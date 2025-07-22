
"""
System-1 Abstraction Layer (S1-AL)
Version: 0.1.0

Extract domain-invariant features from puzzle images:
- Center of Mass (COM)
- Inertia Tensor
- Support Surface polygon

All functions must complete in ≤100 ms/image on target hardware.
"""

__version__ = "0.1.0"

import numpy as np
from shapely.geometry import Polygon

def extract_com(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute 2D center of mass of a binary mask.

    Args:
        mask: Boolean numpy array of shape (H, W).

    Returns:
        (x_com, y_com): Coordinates in pixel space.

    Raises:
        ValueError if mask.sum() == 0.
    """
    if mask.sum() == 0:
        raise ValueError("Empty mask: cannot compute COM.")
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())

def extract_inertia_tensor(mask: np.ndarray) -> np.ndarray:
    """
    Compute the 2×2 inertia tensor of a binary mask.

    Args:
        mask: Boolean numpy array of shape (H, W).

    Returns:
        inertia: 2×2 numpy array [[Ixx, Ixy], [Ixy, Iyy]].
    """
    y, x = np.nonzero(mask)
    x0, y0 = extract_com(mask)
    x_rel, y_rel = x - x0, y - y0
    Ixx = (y_rel**2).sum()
    Iyy = (x_rel**2).sum()
    Ixy = -(x_rel * y_rel).sum()
    return np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=float)

from shapely.geometry.base import BaseGeometry

def extract_support_polygon(mask: np.ndarray) -> BaseGeometry:
    """
    Compute the convex hull of the mask as the support surface.

    Args:
        mask: Boolean numpy array of shape (H, W).

    Returns:
        Polygon: Shapely polygon of the convex hull.
    """
    ys, xs = np.nonzero(mask)
    pts = list(zip(xs, ys))
    return Polygon(pts).convex_hull
