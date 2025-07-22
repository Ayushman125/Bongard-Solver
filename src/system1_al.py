__version__ = "0.1.0"
"""
System-1 Abstraction Layer (S1-AL)
Provides methods to extract domain-invariant features:
 - extract_com(mask) -> (x_com, y_com)
 - extract_inertia_tensor(mask) -> 2×2 numpy array
 - extract_support_polygon(mask) -> shapely Polygon
 - extract_all(mask) -> dict with all above + latency
"""

import time, json
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

# existing free‐functions (extract_com, extract_inertia_tensor, etc.) go here…

_LOG_PATH = Path("logs/s1_al.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

class System1AbstractionLayer:
    def __init__(self):
        pass

    def extract_com(self, mask: np.ndarray) -> tuple[float, float]:
        """Compute center of mass of binary mask."""
        if mask.sum() == 0:
            raise ValueError("Empty mask")
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())

    def extract_inertia_tensor(self, mask: np.ndarray) -> np.ndarray:
        """Compute 2×2 inertia tensor of binary mask."""
        x0, y0 = self.extract_com(mask)
        ys, xs = np.nonzero(mask)
        dx, dy = xs - x0, ys - y0
        Ixx = (dy**2).sum()
        Iyy = (dx**2).sum()
        Ixy = -(dx * dy).sum()
        return np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=float)

    def extract_support_polygon(self, mask: np.ndarray) -> Polygon:
        """Compute convex‐hull support polygon of binary mask."""
        ys, xs = np.nonzero(mask)
        pts = list(zip(xs, ys))
        return Polygon(pts).convex_hull

    def extract_all(self, mask: np.ndarray) -> dict:
        """
        Run all three feature extractors, log JSON event, 
        and return a dict containing:
          - module, timestamp, latency_ms, com, inertia_tensor, support_polygon
        """
        t0 = time.time()
        com = self.extract_com(mask)
        inertia = self.extract_inertia_tensor(mask).tolist()
        poly = list(self.extract_support_polygon(mask).exterior.coords)
        latency = (time.time() - t0) * 1000
        event = {
            "module": "s1_al",
            "timestamp": time.time(),
            "latency_ms": latency,
            "com": com,
            "inertia_tensor": inertia,
            "support_polygon": poly,
        }
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(event) + "\n")
        return event
