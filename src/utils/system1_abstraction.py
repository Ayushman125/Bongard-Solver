__version__ = "0.1.0"
"""
System‐1 Abstraction Layer (S1‐AL)
Version 0.1.0

Public API (all must run ≤100 ms/image on target hardware):
 - extract_com(mask: np.ndarray) -> (float, float)
 - extract_inertia_tensor(mask: np.ndarray) -> np.ndarray
 - extract_support_polygon(mask: np.ndarray) -> shapely.geometry.Polygon

Emits a JSON line to logs/s1_al.jsonl:
  {module:"s1_al", timestamp:…, com:[x,y], inertia:[[...]], polygon_coords:…}
"""
import time, json
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

_LOG_PATH = Path("logs/s1_al.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True)

def extract_com(mask: np.ndarray) -> tuple[float, float]:
    if mask.sum() == 0:
        raise ValueError("Empty mask")
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())

def extract_inertia_tensor(mask: np.ndarray) -> np.ndarray:
    x_com, y_com = extract_com(mask)
    ys, xs = np.nonzero(mask)
    dx, dy = xs - x_com, ys - y_com
    Ixx = (dy**2).sum()
    Iyy = (dx**2).sum()
    Ixy = -(dx * dy).sum()
    return np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=float)

def extract_support_polygon(mask: np.ndarray) -> Polygon:
    ys, xs = np.nonzero(mask)
    pts = list(zip(xs, ys))
    return Polygon(pts).convex_hull

def extract_all(mask: np.ndarray) -> dict:
    """Run all three S1‐AL features and log to file."""
    start = time.time()
    com = extract_com(mask)
    inertia = extract_inertia_tensor(mask)
    poly = extract_support_polygon(mask).exterior.coords[:]
    dt = (time.time() - start) * 1000
    event = {
      "module": "s1_al",
      "timestamp": time.time(),
      "latency_ms": dt,
      "com": com,
      "inertia_tensor": inertia.tolist(),
      "support_polygon": list(poly)
    }
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event
