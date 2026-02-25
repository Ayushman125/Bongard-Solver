__version__ = "0.1.0"
"""
System-1 Abstraction Layer (S1-AL)

Extract domain-invariant features with strict latency SLAs (≤100 ms/image):

  - extract_com(mask: np.ndarray) -> (float, float)
  - extract_inertia_tensor(mask: np.ndarray) -> np.ndarray
  - extract_support_polygon(mask: np.ndarray) -> shapely.geometry.Polygon
  - extract_all(mask: np.ndarray) -> dict

Logs JSON events to logs/s1_al.jsonl.
"""

__version__ = "0.1.0"
import time, json
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

_LOG_PATH = Path("logs/s1_al.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True)

def extract_com(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute the 2D center of mass of a binary mask.
    Args:
        mask: Boolean numpy array of shape (H, W).
    Returns:
        (x_com, y_com): Coordinates in pixel space.
    Raises:
        ValueError: if mask is empty.
    """
    if mask.sum() == 0:
        raise ValueError("Empty mask: cannot compute COM.")
    ys, xs = np.nonzero(mask)
    # shift from array-index to pixel-center coordinates
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)

def extract_inertia_tensor(mask: np.ndarray) -> np.ndarray:
    """
    Compute the 2x2 inertia tensor of a binary mask.
    Args:
        mask: Boolean numpy array of shape (H, W).
    Returns:
        inertia: 2x2 numpy array [[Ixx, Ixy], [Ixy, Iyy]].
    """
    x0, y0 = extract_com(mask)
    ys, xs = np.nonzero(mask)
    dx, dy = xs - x0, ys - y0
    Ixx = (dy**2).sum()
    Iyy = (dx**2).sum()
    Ixy = -(dx * dy).sum()
    return np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=float)

def extract_support_polygon(mask: np.ndarray) -> 'BaseGeometry':
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

def extract_all(mask: np.ndarray) -> dict:
    """
    Run all three S1-AL features and log to file.
    Args:
        mask: Boolean numpy array of shape (H, W).
    Returns:
        event: dict containing all outputs and timing.
    """
    start = time.time()
    com = extract_com(mask)
    inertia = extract_inertia_tensor(mask).tolist()
    poly_coords = list(extract_support_polygon(mask).exterior.coords)
    latency = (time.time() - start) * 1000
    event = {
        "module": "s1_al",
        "timestamp": time.time(),
        "latency_ms": latency,
        "com": com,
        "inertia_tensor": inertia,
        "support_polygon": poly_coords
    }
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event
